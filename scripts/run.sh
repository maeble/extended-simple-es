# Note: this script requires an environment variable: seed=(int)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

export LOG=false

# ----------------------------------------------------------------------------

[ -z "$seed" ] && echo "no variable \$seed defined" && exit 1
export SEED=$seed
./scripts/run/lbf.sh
./scripts/run/rware.sh
./scripts/run/mpe.sh
