# split_cfi.awk
# Splits an assembly file by procedure, inserts LLVM-MCA markers, and writes each to a named output.

BEGIN {
  inside = 0
  wrote = 0
  if (outdir == "") outdir = "out/procs"
  system("mkdir -p \"" outdir "\"")

  # Precompile common regexes
  re_cfi_start = "^[[:space:]]*\\.cfi_startproc([[:space:]]|$)"
  re_cfi_end   = "^[[:space:]]*\\.cfi_endproc([[:space:]]|$)"
  re_label     = "^[[:space:]]*\\.([A-Za-z0-9_]+):[[:space:]]*$"
  # Match: [start or space] b OR b.<cond>  then spaces then .TargetLabel
  re_branch    = "(^|[[:space:]])b(\\.[A-Za-z]+)?[[:space:]]+\\.([A-Za-z0-9_]+)"

  # Parse the space-separated list of output file basenames
  if (outfiles == "") {
    print "split_cfi.awk: please provide -v outfiles=\"name1 name2 ...\"" > "/dev/stderr"
    exit 2
  }
  gsub(/,/, " ", outfiles)
  # squeeze repeated whitespace
  while (sub(/[[:space:]]{2,}/, " ", outfiles)) {}
  num_outs = split(outfiles, outnames, /[[:space:]]+/)
  next_idx = 1
}

/^[[:space:]]*\.cfi_startproc([[:space:]]|$)/ {
  if (next_idx > num_outs) {
    printf "split_cfi.awk: not enough names in outfiles (need at least %d; got %d)\n", next_idx, num_outs > "/dev/stderr"
    exit 2
  }
  outfile = sprintf("%s/%s.s", outdir, outnames[next_idx++])
  inside = 1
  wrote = 1
  delete lines
  line_count = 0
  lines[++line_count] = $0
  next
}

inside {
  lines[++line_count] = $0

  if ($0 ~ re_cfi_end) {
    largest_start = 0
    largest_end = 0
    largest_size = -1  # so size 0 still beats "none"

    # scan labels
    for (i = 1; i <= line_count; i++) {
      line = lines[i]
      # Is this a label?
      if (match(line, re_label, mlabel)) {
        label_name = mlabel[1]

        # Search forward until next label
        for (j = i + 1; j <= line_count; j++) {
          if (match(lines[j], re_label)) break

          # Look for branch to this same label
          if (match(lines[j], re_branch, mbr)) {
            # mbr[3] is the captured target label (after the dot)
            target = mbr[3]
            if (target == label_name) {
              # "largest" = most newlines between label and branch
              loop_size = (j - i - 1)
              if (loop_size > largest_size) {
                largest_size = loop_size
                largest_start = i
                largest_end = j
              }
              break
            }
          }
        }
      }
    }

    # Write with markers
    for (i = 1; i <= line_count; i++) {
      if (i == largest_start && largest_size >= 0) {
        print "# LLVM-MCA-BEGIN" > outfile
      }
      print lines[i] > outfile
      if (i == largest_end && largest_size >= 0) {
        print "# LLVM-MCA-END" > outfile
      }
    }

    close(outfile)
    inside = 0
  }
  next
}

END {
  if (!wrote) {
    print "split_cfi.awk: no .cfi_startproc blocks found" > "/dev/stderr"
    exit 1
  }
  if (next_idx <= num_outs) {
    # Extra names provided; not fatal, just warn.
    printf "split_cfi.awk: warning: %d unused name(s) in outfiles\n", (num_outs - next_idx + 1) > "/dev/stderr"
  }
}
