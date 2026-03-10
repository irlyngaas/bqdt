import os
import time
import math
import heapq
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 尝试导入 CuPy
try:
    import cupy as cp
    from cupyx.scipy import ndimage
    from cupyx.scipy.interpolate import RegularGridInterpolator
    GPU = True
    xp = cp
    print("[System] Using CuPy (GPU).")
except ImportError:
    cp = None
    xp = np
    GPU = False
    print("[System] CuPy not available, using NumPy (CPU).")

# -------------------------
# 辅助函数
# -------------------------
def pad_to_multiple(arr, block):
    """确保数组尺寸能被 block 整除"""
    h, w, c = arr.shape
    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block
    if pad_h == 0 and pad_w == 0:
        return arr, (h, w)
    # 使用 xp.pad (GPU if avail)
    padded = xp.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge') 
    return padded, (h, w)

def ensure_cpu(arr):
    """辅助函数：确保数据在 CPU 上 (用于可视化或循环控制)"""
    if GPU and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr

# -------------------------
# 1. 优化后的 GPU 合并器 (Our Merge)
# -------------------------
class GPUQuadTreeMerger:
    def __init__(self, image_path, min_size=2, patch_size=(16,16)):
        self.min_size = min_size
        self.patch_size = patch_size
        self.pil = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.pil.size
        
        # 原始数据转为 array
        arr = np.array(self.pil)
        # 移至 GPU
        self.img = xp.array(arr) if GPU else arr
        self.H, self.W = self.img.shape[0], self.img.shape[1]
        
        # 计算层级
        max_blocks = min(self.H // self.min_size, self.W // self.min_size)
        self.max_level = int(math.floor(math.log2(max_blocks))) if max_blocks >= 1 else 0
        
        final_block = self.min_size * (2 ** self.max_level)
        self.img, (self.orig_h_pad, self.orig_w_pad) = pad_to_multiple(self.img, final_block)
        self.Hp, self.Wp = self.img.shape[0], self.img.shape[1]

        print(f"[Info] Level: 0-{self.max_level}, Padded Size: {self.Wp}x{self.Hp}")

        # 存储统计数据 (保持在 xp/GPU 上)
        self.means = [] 
        self.errors = [] 
        self.level_shapes = [] 
        self.merge_costs = [None] * (self.max_level + 1)
        
        # 计时 Precompute
        if GPU: cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        self._precompute_all_levels()
        if GPU: cp.cuda.Device().synchronize()
        print(f"[Precompute] Time: {time.perf_counter() - t0:.4f}s")

    def _compute_level_stats(self, block_size):
        img = self.img
        Hp, Wp = img.shape[0], img.shape[1]
        gh, gw = Hp // block_size, Wp // block_size
        
        # (gh, block, gw, block, 3) -> (gh, gw, block, block, 3)
        reshaped = img.reshape(gh, block_size, gw, block_size, 3).transpose(0, 2, 1, 3, 4)
        
        # GPU 计算
        means = xp.mean(reshaped, axis=(2, 3))
        # Var 计算
        vars_ = xp.var(reshaped, axis=(2, 3))
        sse = xp.sum(vars_, axis=2) * (block_size * block_size)
        
        # 关键修改：**不调用 .get()**，保持在 GPU 上
        return means.astype(xp.uint8), sse

    def _precompute_all_levels(self):
        for l in range(self.max_level + 1):
            bs = self.min_size * (2 ** l)
            means_l, sse_l = self._compute_level_stats(bs)
            self.means.append(means_l)
            self.errors.append(sse_l)
            self.level_shapes.append(sse_l.shape)

        # 计算合并代价 (Parent Error - Sum(Child Errors))
        # 全程 GPU 运算
        for level in range(1, self.max_level + 1):
            child_err = self.errors[level - 1]
            Hp, Wp = child_err.shape[0] // 2, child_err.shape[1] // 2
            
            # Reshape 4 children to sum
            c_reshaped = child_err.reshape(Hp, 2, Wp, 2)
            sum_children = xp.sum(c_reshaped, axis=(1, 3))
            
            parent_err = self.errors[level]
            cost = parent_err - sum_children
            self.merge_costs[level] = cost

    def run_merge(self, start_leaves=None, end_leaves=16384, batch_k=None, visualize_path="result_optimized_merge.png"):
        total_L0 = self.level_shapes[0][0] * self.level_shapes[0][1]
        if start_leaves is None: start_leaves = total_L0
        
        # alive 数组保持在 GPU (xp)
        alive = [xp.zeros(shape, dtype=bool) for shape in self.level_shapes]
        alive[0][:] = True
        
        # 需要转为 int (Python scalar) 进行比较
        current_leaves = int(xp.sum(alive[0]))
        print(f"\n[Merge] Start: {current_leaves} -> End: {end_leaves}")

        if GPU: cp.cuda.Device().synchronize()
        t_start = time.perf_counter()
        
        iter_count = 0
        
        # 动态 Batch 策略参数
        # 每次尝试合并当前存活节点的 15%
        batch_ratio = 0.15  
        min_batch = 4096
        max_batch = 500000 # 上限 50万，防止 GPU 排序 OOM
        
        while current_leaves > end_leaves:
            iter_count += 1
            
            # --- 动态计算本轮 Batch Size (递减策略) ---
            # 随着 current_leaves 减少，batch_k 自动减少，从粗犷变得精细
            target_batch = int(current_leaves * batch_ratio)
            batch_k = max(min_batch, min(target_batch, max_batch))
            
            # 如果剩下的合并量很少，直接尝试一次性完成
            if (current_leaves - end_leaves) < batch_k:
                 batch_k = max(min_batch, current_leaves - end_leaves + 1000)

            candidates_cost = []
            candidates_meta = [] 
            
            # 1. 收集阶段 (GPU Kernel Launches)
            for level in range(1, self.max_level + 1):
                Hp, Wp = self.level_shapes[level]
                
                # View children status
                child_alive_view = alive[level-1].reshape(Hp, 2, Wp, 2)
                # Check if all 4 children are alive
                can_merge = xp.all(child_alive_view, axis=(1, 3))
                
                if not xp.any(can_merge): continue
                
                costs = self.merge_costs[level]
                
                # 获取索引 (GPU array)
                valid_indices = xp.where(can_merge) 
                valid_costs = costs[valid_indices]
                num_valid = valid_costs.shape[0]
                
                candidates_cost.append(valid_costs)
                
                # 构建元数据 [level, r, c] on GPU
                level_arr = xp.full(num_valid, level, dtype=xp.int32)
                meta_data = xp.column_stack((level_arr, valid_indices[0], valid_indices[1]))
                candidates_meta.append(meta_data)

            if not candidates_cost:
                break
                
            # 2. 排序与筛选 (GPU Sort)
            all_costs = xp.concatenate(candidates_cost)
            all_meta = xp.concatenate(candidates_meta)
            
            current_candidates_count = all_costs.shape[0]
            k = int(min(batch_k, current_candidates_count)) 
            
            # H100 Optimization: argpartition 
            top_k_indices_unsorted = xp.argpartition(all_costs, k-1)[:k]
            
            subset_costs = all_costs[top_k_indices_unsorted]
            sorted_local_idx = xp.argsort(subset_costs)
            
            final_indices = top_k_indices_unsorted[sorted_local_idx]
            selected_meta = all_meta[final_indices]
            
            # 3. 应用合并 (GPU Indexing)
            merges_count = 0
            level_break = False
            for lvl in range(1, self.max_level + 1):
                # GPU Mask
                mask = (selected_meta[:, 0] == lvl)
                
                rows = selected_meta[mask, 1] 
                cols = selected_meta[mask, 2] 

                mc = merges_count + rows.shape[0]
                cl = current_leaves -3*mc
                if cl < end_leaves:
                    cl = current_leaves -3*merges_count
                    partial_level = lvl
                    level_break = True
                    partial_index = int((cl - end_leaves)/3)
                    break
                
                if rows.size == 0: continue

                # 更新 Alive 表
                alive[lvl][rows, cols] = True
                
                tgt = alive[lvl-1]
                r2 = rows * 2
                c2 = cols * 2
                tgt[r2, c2] = False
                tgt[r2+1, c2] = False
                tgt[r2, c2+1] = False
                tgt[r2+1, c2+1] = False
                
                merges_count += rows.shape[0]
                cl = current_leaves -3*mc
                if cl == end_leaves:
                    break
            if level_break: #Perform a partial level break to get exact amount of end leaves
                mask = (selected_meta[:, 0] == partial_level)
                rows = selected_meta[mask, 1] 
                rows = rows[:partial_index]
                cols = selected_meta[mask, 2] 
                cols = cols[:partial_index]
                alive[lvl][rows, cols] = True
                
                tgt = alive[lvl-1]
                r2 = rows * 2
                c2 = cols * 2
                tgt[r2, c2] = False
                tgt[r2+1, c2] = False
                tgt[r2, c2+1] = False
                tgt[r2+1, c2+1] = False
                
                merges_count += rows.shape[0]

            current_leaves -= 3 * merges_count
            
            # 不再需要手动的 batch 增长策略，自动递减
        print("Target End Leaves: ", end_leaves)
        print("Actual End Leaves: ", current_leaves)

        if GPU: cp.cuda.Device().synchronize()
        t_total = time.perf_counter() - t_start
        print(f"[Merge Done] Time: {t_total:.4f}s (Iterations: {iter_count})")
        self._save_visualization(alive, visualize_path)
        border_mask = self._border_mask(alive, end_leaves)
        regions = self._detect_regions(border_mask)
        serialized_input = self._serialize(regions)
        return t_total

    def _save_visualization(self, alive, filename):
        print("[Visualization] Converting GPU data to CPU for image generation...")
        
        # 1. 获取原始图像数据作为底图
        img_cpu = ensure_cpu(self.img).astype(np.uint8)
        out = Image.fromarray(img_cpu)
        draw = ImageDraw.Draw(out)
        
        grid_color = (255, 255, 255)
        
        # 从高层向下绘制
        for level in reversed(range(len(alive))):
            bs = self.min_size * (2 ** level)
            
            if bs <= 2: continue
            
            alive_cpu = ensure_cpu(alive[level])
            coords = np.argwhere(alive_cpu)
            
            if len(coords) == 0: continue
            
            for r, c in coords:
                x0, y0 = c * bs, r * bs
                draw.rectangle([x0, y0, x0 + bs, y0 + bs], fill=None, outline=grid_color)
        
        out.crop((0, 0, self.orig_w, self.orig_h)).save(filename)
        print(f"[Visualization] Saved to {filename}")

    def _border_mask(self, alive, end_leaves, save_border_mask=True):
    #On AMD cupy has bug when size > 4096, should be tested on NVIDIA GPU
        out = xp.full((self.orig_h_pad, self.orig_w_pad), False)
        for level in reversed(range(len(alive))):
            bs = self.min_size * (2 ** level)
            
            if bs < 2: continue
            
            coords = xp.argwhere(alive[level])
            
            if len(coords) == 0: 
                continue
            else: 
                for r, c in coords:
                    x0, y0 = r * bs, c * bs

                    if x0 == 0:
                        out[x0:x0+1, y0:y0+bs] = True
                    else:
                        out[x0-1:x0, y0:y0+bs] = True
                    out[x0+bs-1:x0+bs, y0:y0+bs] = True
                    if y0 == 0:
                        out[x0:x0+bs, y0:y0+1] = True
                    else:
                        out[x0:x0+bs, y0-1:y0] = True
                    out[x0:x0+bs, y0+bs-1:y0+bs] = True
        if save_border_mask:
            out_cpu = xp.asnumpy(out)
            out_cpu = out_cpu.astype(np.uint8)
            plt.imshow(out_cpu)
            plt.colorbar()
            plt.savefig("border_mask.png")
        return out

    def _detect_regions(self, border_mask, print_regions=False):
        interior = ~border_mask
        labeled, num_features = ndimage.label(interior)
        regions = []
        if print_regions:
            print("x0, x1, y0, y1, h, w, cx, cy")
        for region_id in range(1, num_features + 1):
            ys, xs = xp.where(labeled == region_id)

            x0, x1 = int(xs.min()), int(xs.max())
            if x0 == 1:
                x0 = x0-1
            x1 = x1+1

            y0, y1 = int(ys.min()), int(ys.max())
            if y0 == 1:
                y0 = y0-1
            y1 = y1+1
            w = x1 - x0 + 1
            h = y1 - y0 + 1
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            if print_regions:
                print(x0, x1, y0, y1, h, w, cx, cy)
            regions.append({
                            "x0": y0,
                            "x1": y1,
                            "y0": x0,
                            "y1": x1,
                            "height": h,
                            "width": w,
                            "cx": cx,
                            "cy": cy,
            })

        return regions

    def _serialize(self, regions):
        h2,w2 = self.patch_size
        c2 = self.img.shape[2]
        serialize = []
        for i in range(len(regions)):
            h1, w1, = regions[i]["height"], regions[i]["width"]
            assert h1==w1, "Need squared input."

            h1_ = xp.linspace(0,h1,h1)
            w1_ = xp.linspace(0,w1,w1)
            interp_fct_list = []
            for j in range(c2):
                interp_fct_list.append(RegularGridInterpolator(points=[h1_,w1_], values=self.img[regions[i]["x0"]:regions[i]["x1"]+1,regions[i]["y0"]:regions[i]["y1"]+1,j]))

            patch_ = xp.zeros([h2,w2,c2])
            h2_ = xp.linspace(0,h1,h2)
            w2_ = xp.linspace(0,w1,w2)
            H2_, W2_ = xp.meshgrid(h2_, w2_, indexing='ij')
            query_points = xp.vstack([H2_.ravel(),W2_.ravel()]).T
            for j in range(c2):
                patch_[:,:,j] = interp_fct_list[j](query_points).reshape(H2_.shape)
            serialize.append(patch_)
        return serialize

# -------------------------
# 2. 优化后的 Heap Baseline (Optimized Heap)
# -------------------------
class HeapSplitBaseline:
    """
    使用 Priority Queue (Heap) 的基线算法。
    复杂度: O(N log N)
    """
    def __init__(self, image_path):
        self.pil = Image.open(image_path).convert("RGB")
        self.img_arr = np.array(self.pil)
        self.H, self.W = self.img_arr.shape[:2]
        
    def get_error_and_color(self, x0, y0, x1, y1):
        patch = self.img_arr[y0:y1, x0:x1]
        if patch.size == 0: return 0, (0,0,0)
        var = np.var(patch, axis=(0, 1))
        err = np.sum(var) * patch.size
        mean = np.mean(patch, axis=(0, 1)).astype(np.uint8)
        return err, tuple(mean)

    def run(self, target_leaves=16384, output="result_heap_split.png"):
        print(f"\n[Heap Baseline] Start Split -> {target_leaves}")
        
        root_err, root_col = self.get_error_and_color(0, 0, self.W, self.H)
        heap = [(-root_err, 0, 0, self.W, self.H, root_col)]
        
        start_t = time.perf_counter()
        
        while len(heap) < target_leaves:
            neg_err, x0, y0, x1, y1, col = heapq.heappop(heap)
            w, h = x1 - x0, y1 - y0
            if w <= 1 or h <= 1:
                heapq.heappush(heap, (neg_err, x0, y0, x1, y1, col))
                break
                
            mx, my = x0 + w//2, y0 + h//2
            sub_rects = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
            
            for rx0, ry0, rx1, ry1 in sub_rects:
                if rx1 > rx0 and ry1 > ry0:
                    ne, nc = self.get_error_and_color(rx0, ry0, rx1, ry1)
                    heapq.heappush(heap, (-ne, rx0, ry0, rx1, ry1, nc))
                    
        total_t = time.perf_counter() - start_t
        print(f"[Heap Baseline] Done in {total_t:.4f}s")
        self._draw(heap, output)
        return total_t

    def _draw(self, heap, filename):
        out = Image.fromarray(self.img_arr)
        draw = ImageDraw.Draw(out)
        grid_color = (255, 255, 255)
        
        for item in heap:
            _, x0, y0, x1, y1, col = item
            w = x1 - x0
            if w <= 2: continue
            draw.rectangle([x0, y0, x1, y1], fill=None, outline=grid_color)
        out.save(filename)

# -------------------------
# 3. APT 原文 Baseline (APT Baseline)
# -------------------------
class APTSplitBaseline:
    """
    复刻 Adaptive Patching 原文中的实现：
    使用 List 存储叶子节点，每次遍历 List 寻找 Max Error。
    复杂度: O(N^2)
    """
    def __init__(self, image_path):
        self.pil = Image.open(image_path).convert("RGB")
        self.img_arr = np.array(self.pil)
        self.H, self.W = self.img_arr.shape[:2]
        
    def get_error_and_color(self, x0, y0, x1, y1):
        patch = self.img_arr[y0:y1, x0:x1]
        if patch.size == 0: return 0, (0,0,0)
        var = np.var(patch, axis=(0, 1))
        err = np.sum(var) * patch.size
        mean = np.mean(patch, axis=(0, 1)).astype(np.uint8)
        return err, tuple(mean)

    def run(self, target_leaves=16384, output="result_apt_split.png"):
        print(f"\n[APT Baseline] Start Split (List Max) -> {target_leaves}")
        print("[APT Baseline] Warning: This O(N^2) method is expected to be very slow for large N.")
        
        root_err, root_col = self.get_error_and_color(0, 0, self.W, self.H)
        # 使用普通的 List, 不使用 Heap
        leaves = [(root_err, 0, 0, self.W, self.H, root_col)]
        
        start_t = time.perf_counter()
        
        # 循环直到达到目标数量
        while len(leaves) < target_leaves:
            # --- Inefficient Scan: O(N) to find max ---
            max_idx = -1
            max_err = -1.0
            
            # 手动遍历或使用 python max()，这里使用 enumerate 模拟线性扫描
            for i, (err, _, _, _, _, _) in enumerate(leaves):
                if err > max_err:
                    max_err = err
                    max_idx = i
            
            if max_idx == -1: 
                break # Should not happen

            # --- Pop & Split ---
            # pop(idx) 也是 O(N) 操作
            err, x0, y0, x1, y1, col = leaves.pop(max_idx)
            
            w, h = x1 - x0, y1 - y0
            if w <= 1 or h <= 1:
                # 无法再分，放回去并将 error 设为 -1 防止再次被选中
                leaves.append((-1.0, x0, y0, x1, y1, col))
                continue
                
            mx, my = x0 + w//2, y0 + h//2
            sub_rects = [(x0, y0, mx, my), (mx, y0, x1, my), (x0, my, mx, y1), (mx, my, x1, y1)]
            
            for rx0, ry0, rx1, ry1 in sub_rects:
                if rx1 > rx0 and ry1 > ry0:
                    ne, nc = self.get_error_and_color(rx0, ry0, rx1, ry1)
                    leaves.append((ne, rx0, ry0, rx1, ry1, nc))
            
            # 打印进度防止以为卡死
            if len(leaves) % 1000 == 0:
                elapsed = time.perf_counter() - start_t
                print(f"\r[APT Baseline] Leaves: {len(leaves)}/{target_leaves}, Time: {elapsed:.2f}s", end="")

        total_t = time.perf_counter() - start_t
        print(f"\n[APT Baseline] Done in {total_t:.4f}s")
        self._draw(leaves, output)
        return total_t

    def _draw(self, leaves, filename):
        out = Image.fromarray(self.img_arr)
        draw = ImageDraw.Draw(out)
        grid_color = (255, 255, 255)
        
        for item in leaves:
            _, x0, y0, x1, y1, col = item
            w = x1 - x0
            if w <= 2: continue
            draw.rectangle([x0, y0, x1, y1], fill=None, outline=grid_color)
        out.save(filename)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # 请确保此处图片路径正确
    img_path = "bqdt-test.jpg"
    img = Image.open(img_path)
    new_size = (2048,2048)
    resized_img = img.resize(new_size, Image.LANCZOS)
    resized_img.save("bqdt-test_resize.jpg")
    img_path = "bqdt-test_resize.jpg"
    
    # 设定目标叶子节点数量
    #TARGET_LEAVES needs to be some number such that n is a Natural Number and 3*n+1, here n=43690 = 131071
    #n = 43690
    n = 100
    TARGET_LEAVES = 3*n + 1

    #This number doesn't work since it doesn't 
    #TARGET_LEAVES = 131072
    
    
    print("="*60)
    print(f"Benchmark Start. Target Leaves: {TARGET_LEAVES}")
    print("="*60)

    # 1. Run Optimized Merge (Our Method)
    #patch_size is the uniform patch size for serialize
    merger = GPUQuadTreeMerger(img_path, min_size=1, patch_size=(16,16))
    t_merge = merger.run_merge(start_leaves=None, end_leaves=TARGET_LEAVES, batch_k=8192, visualize_path="result_merge.png")

    ## 2. Run Optimized Baseline (Heap)
    #heap_base = HeapSplitBaseline(img_path)
    #t_heap = heap_base.run(target_leaves=TARGET_LEAVES, output="result_heap.png")

    ## 3. Run APT Baseline (Naive List)
    ## 注意: 在 13万 节点下，O(N^2) 会非常慢。
    #apt_base = APTSplitBaseline(img_path)
    #t_apt = apt_base.run(target_leaves=TARGET_LEAVES, output="result_apt.png")
    #
    #print("\n" + "="*60)
    #print("Final Comparison Results (Time in Seconds)")
    #print("="*60)
    print(f"1. Our Merge (GPU Vectorized) : {t_merge:.4f} s")
    #print(f"2. Optimized Baseline (Heap)  : {t_heap:.4f} s")
    #print(f"3. APT Baseline (Naive List)  : {t_apt:.4f} s")
    #print("="*60)
