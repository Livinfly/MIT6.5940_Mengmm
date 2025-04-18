# Lab5

## Problems

> When running `./evaluate.sh reference`, I encountered the error `error: 'int8_t' has not been declared` and solve it by adding `#include <cstdint>` or `<stdint.h>` in `kernels/quantizer.cc`.

> When running `chat.exe`, I encountered the error `terminate called after throwing an instance of 'char const*'` and found it was caused by insufficient memory, so ensure you has adequate memory.

>   download_model.py add:
>
>   if os.path.isfile(filepath):  # Avoid redownloading
>
>   ​      print(f"File already exists: {filepath}")
>
>   ​    else:
>
>   ​      _download_file(url, filepath)

## Evaluate

The supported argument:

- reference
- loop_unrolling
- multithreading
- simd_programming
- multithreading_loop_unrolling
- all_techniques

## Results


### Reference

```sh
-------- Sanity check of reference implementation: Passed! --------
Section, Total time(ms), Average time(ms), Count, GOPs
reference, 3494.197021, 349.419006, 10, 0.750227

All tests completed!
```

### loop_unrolling


### multithreading


### simd_programming


### multithreading_loop_unrolling


### all_techniques


## Instructions to implement each technique:

1. Complete the provided starter code by adding the necessary code segments, which are marked with     "TODO" in the comments. You **only** need to write the code for your device, i.e., QM_x86 for x86 CPUs and QM_ARM for ARM CPUs. It’s recommended to consult the reference implementation (***reference.cc\***) and work on the templates in the following sequence:
2. 1. Loop Unrolling (loop_unrolling.cc)
   2. Multithreading (multithreading.cc)
   3. SIMD Programming (simd_programming.cc)
   4. Multithreading with Loop Unrolling (multithreading_loop_unrolling.cc)
   5. Combination of All Techniques (all_techniques.cc)
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
