#!/bin/bash
# Fortran check: a tiny Fortran extension builds with f2py (meson backend) and imports.
# Stands in for the SPAM Fortran build.
set -euo pipefail

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"

cat > gtcadd.f90 <<'EOF'
subroutine add(a, b, c)
  real(8), intent(in) :: a, b
  real(8), intent(out) :: c
  c = a + b
end subroutine add
EOF

echo "gfortran: $(gfortran --version | head -1)"
if ! python -m numpy.f2py -c gtcadd.f90 -m gtcadd > build.log 2>&1; then
    cat build.log
    echo "FAIL fortran"
    exit 1
fi
python -c "import gtcadd; r = gtcadd.add(2.0, 3.0); assert abs(r - 5.0) < 1e-12, r; print('add(2, 3) =', r)"
echo "PASS fortran"
