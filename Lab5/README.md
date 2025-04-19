# Lab5

Implements are in `starter_coder`.

## Problems

-   When running `./evaluate.sh reference`, I encountered the error `error: 'int8_t' has not been declared` and solve it by adding `#include <cstdint>` or `<stdint.h>` in `kernels/quantizer.cc`.

-   When running `chat.exe`, I encountered the error `terminate called after throwing an instance of 'char const*'` and found it was caused by insufficient memory, so ensure you has adequate memory.

-   download_model.py add:

    ```python
    if os.path.isfile(filepath):  # Avoid redownloading
    	print(f"File already exists: {filepath}")
    else:
        _download_file(url, filepath)
    ```

-   Some typo fix:

    In starter_code, `order of weights with QM_x86`, `QM_ARM order`=> `QM_x86 order`

    In `QM_x86`, use `w_de_32` instead of `w_de_16` to fit order of weights with QM_x86 better.

-   params about Quantization like zero_point are simpified, using 8.

## System Info

-   **CPU**: AMD Ryzen 7 6800H with Radeon Graphics @ 3.20GHz
-   **GPU**: AMD Radeon(TM) Graphics
-   **Memory**: 16GB

## Evaluate

The supported argument:

- reference
- loop_unrolling
- multithreading
- simd_programming
- multithreading_loop_unrolling
- all_techniques

## Results

|                    Section                     | Total time(ms) | Average time(ms) | Count | GOPs      |
| :--------------------------------------------: | :------------- | ---------------- | ----- | :-------- |
|                   reference                    | 3494.197021    | 349.419006       | 10    | 0.750227  |
|                 loop_unrolling                 | 2645.466064    | 264.545990       | 10    | 0.990918  |
|        multithreading (thread_num = 4)         | 956.875977     | 95.686996        | 10    | 2.739582  |
|                simd_programming                | 2293.267090    | 229.326004       | 10    | 1.143103  |
| multithreading_loop_unrolling (thread_num = 4) | 699.916016     | 69.990997        | 10    | 3.745364  |
|        all_techniques (thread_num = 8)         | 196.772995     | 19.677000        | 10    | 13.322153 |
|        all_techniques (thread_num = 4)         | 369.049011     | 36.903999        | 10    | 7.103230  |

**These three optimization approaches exhibit multiplicative speedup when applied concurrently.**



### reference

```sh
# ./evaluate.sh reference
-------- Sanity check of reference implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
reference, 3494.197021, 349.419006, 10, 0.750227

All tests completed!
```

```sh
# chat.exe
Using model: LLaMA_7B_2_chat
Using LLaMA's default data format: INT4
Loading model... Finished!
USER:
ASSISTANT:
How are you today?
Section, Total time(ms), Average time(ms), Count, GOPs
Inference latency, 106339.968750, 17723.328125, 6, N/A
USER:
ASSISTANT:
How are you today?
Section, Total time(ms), Average time(ms), Count, GOPs
Inference latency, 105717.156250, 17619.525391, 6, N/A
USER:
ASSISTANT:
How can I help you today?
Section, Total time(ms), Average time(ms), Count, GOPs
Inference latency, 141133.062500, 17641.632812, 8, N/A
USER:
```

### loop_unrolling

```sh
# ./evaluate.sh loop_unrolling
-------- Sanity check of loop_unrolling implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
loop_unrolling, 2645.466064, 264.545990, 10, 0.990918

All tests completed!
```

```sh
# chat.exe

```

### multithreading

```sh
# ./evaluate.sh multithreading
-------- Sanity check of multithreading implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading, 956.875977, 95.686996, 10, 2.739582

All tests completed!
```

```sh
# chat.exe

```

### simd_programming

```sh
# ./evaluate.sh simd_programming
-------- Sanity check of simd_programming implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
simd_programming, 2293.267090, 229.326004, 10, 1.143103

All tests completed!
```

```sh
# chat.exe

```

### multithreading_loop_unrolling

```sh
# ./evaluate.sh multithreading_loop_unrolling
-------- Sanity check of multithreading_loop_unrolling implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading_loop_unrolling, 699.916016, 69.990997, 10, 3.745364

All tests completed!
```

```sh
# chat.exe

```

### all_techniques

```sh
# ./evaluate.sh all_techniques
-------- Sanity check of all_techniques implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
all_techniques, 196.772995, 19.677000, 10, 13.322153

All tests completed!
```

```sh
# chat.exe

```


## Instructions to implement each technique:

1. Complete the provided starter code by adding the necessary code segments, which are marked with     "TODO" in the comments. You **only** need to write the code for your device, i.e., QM_x86 for x86 CPUs and QM_ARM for ARM CPUs. It’s recommended to consult the reference implementation (***reference.cc\***) and work on the templates in the following sequence:

    1.   Loop Unrolling (loop_unrolling.cc)

    2.   Multithreading (multithreading.cc)

    3.   SIMD Programming (simd_programming.cc)
    4.   Multithreading with Loop Unrolling (multithreading_loop_unrolling.cc)
    5.   Combination of All Techniques (all_techniques.cc)

3. Evaluate the correctness of the implementation using the evaluation script.

## Grading Policy:

- Total Points: 120 points (20 points for each optimization implementation and 20 points for bonus).

  - Correctness: 15 points. This will be based on the output of the evaluation script.
  - Performance Reporting: 5 points. Measure the performance improvement achieved by each technique on your computer and why it improves the performance.
  - Bonus: Any optimization techniques on your mind? Try to implement them to improve the performance further! If you can further improve the performance compared to the optimized kernel in TinyChatEngine, you can get bonus points here! Each percent of performance speedup equals one point up to 20 points.
- Submission:

  - Report: Please write a report (form) that includes your code and the performance improvement for each starter code.
    - Report template: https://docs.google.com/document/d/17Z_ab8EhDvjcigLXdDqMqd2LTVsZ4CnpOYNkRTrnTmU/edit?usp=sharing
- Code: Use `git diff` to generate a patch for your implementation. We will use this patch to test the correctness of your code. Please name your patch as `{studentID}-{ISA}.patch` where {ISA} should be one of x86 and ARM, depending on your computer.
