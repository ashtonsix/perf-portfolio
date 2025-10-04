# split_cfi.awk
# Usage: awk -v outdir="out/procs" -f split_cfi.awk out/bytepack.s
# Writes out/procs/bytepack_1_pack.s, bytepack_1_unpack.s, bytepack_2_pack.s, ...

BEGIN {
  inside = 0
  n = 1
  wrote = 0
  if (outdir == "") outdir = "out/procs"
  system("mkdir -p \"" outdir "\"")
}

# Start a new output file on .cfi_startproc
/^[[:space:]]*\.cfi_startproc([[:space:]]|$)/ {
  n++
  base    = int(n / 2)
  suffix  = (n % 2 == 0) ? "pack" : "unpack"
  outfile = sprintf("%s/bytepack_%d_%s.s", outdir, base, suffix)

  inside = 1
  wrote = 1
  print $0 > outfile
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

END {
  # Exit non-zero so make fails if nothing was produced.
  if (!wrote) {
    print "split_cfi.awk: no .cfi_startproc blocks found" > "/dev/stderr"
    exit 1
  }
}
