# split_delta.awk
# Splits assembly file into separate files per function

BEGIN {
  inside = 0
  func_name = ""
  if (outdir == "") outdir = "out/asm"
  if (prefix == "") prefix = "func"
  system("mkdir -p \"" outdir "\"")
}

# Capture function name from .type directive
/\.type.*,@function/ {
  match($0, /\.type[[:space:]]+([^,]+)/, arr)
  if (arr[1] != "") {
    func_label = arr[1]
    
    # Extract function names
    if (match(func_label, /delta_D1_W32_baseline_naive/)) {
      func_name = "delta_D1_W32_baseline_naive"
    } else if (match(func_label, /prefix_D1_W32_baseline_naive/)) {
      func_name = "prefix_D1_W32_baseline_naive"
    } else if (match(func_label, /delta_D1_W32_baseline_SIMD_x86/)) {
      func_name = "delta_D1_W32_baseline_SIMD_x86"
    } else if (match(func_label, /prefix_D1_W32_baseline_SIMD_x86/)) {
      func_name = "prefix_D1_W32_baseline_SIMD_x86"
    } else if (match(func_label, /delta_D1_W32_baseline_SIMD_ARM/)) {
      func_name = "delta_D1_W32_baseline_SIMD_ARM"
    } else if (match(func_label, /prefix_D1_W32_baseline_SIMD_ARM/)) {
      func_name = "prefix_D1_W32_baseline_SIMD_ARM"
    } else if (match(func_label, /prefix_D1_W32_unrolled/)) {
      func_name = "prefix_D1_W32_unrolled"
    } else if (match(func_label, /prefix_D1_W32_pipelined/)) {
      func_name = "prefix_D1_W32_pipelined"
    } else if (match(func_label, /prefix_D1_W32_transpose/)) {
      func_name = "prefix_D1_W32_transpose"
    }
  }
}

# Start a new output file on .cfi_startproc
/^[[:space:]]*\.cfi_startproc([[:space:]]|$)/ {
  if (func_name != "") {
    outfile = sprintf("%s/%s.s", outdir, func_name)
    inside = 1
    print $0 > outfile
    func_name = ""
  }
  next
}

# Copy lines while inside, stop on .cfi_endproc (inclusive)
inside {
  print $0 > outfile
  if ($0 ~ /^[[:space:]]*\.cfi_endproc([[:space:]]|$)/) {
    close(outfile)
    inside = 0
  }
  next
}
