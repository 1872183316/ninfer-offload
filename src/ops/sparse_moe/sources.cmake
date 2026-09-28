target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/sparse_moe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_rowsplit_dot.cpp"
)
# Host expert kernels require Haswell-class x86 (AVX2, FMA, F16C, BMI2); support is checked at runtime.
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/host/host_rowsplit_dot.cpp"
  PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mbmi2")
