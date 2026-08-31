1. Vector addition
       ✓

2. Reduction
       ↓
   shared memory
   synchronization
   race conditions

3. Naive matrix multiplication
       ↓
   2D thread indexing

4. Tiled matrix multiplication
       ↓
   shared memory
   data reuse
   memory bandwidth

5. Optimize tiled matmul
       ↓
   coalescing
   bank conflicts
   occupancy
   register pressure

6. Tensor-core matmul
       ↓
   WMMA / MMA instructions

===
more shared memory / block
        ↓
potentially fewer resident blocks
        ↓
potentially fewer resident warps
        ↓
potentially worse latency hiding

BUT

more shared memory
        ↓
more data reuse / cooperation
        ↓
potentially much less global-memory traffic
        ↓
potentially much faster kernel