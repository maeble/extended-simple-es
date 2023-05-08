BLUE='\033[0;34m'
NC='\033[0m'

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GEN_NUM=20500 # *100 time_steps per generation = 20050000 total timesteps

# ----------------------------------------------------------------------------

args=()
[ ! -z "$GEN_NUM" ] && args+=( "--generation-num" ) && args+=( "$GEN_NUM" ) 
[ $LOG == true ] && args+=( '--log' )
[ ! -z "$SEED"] && args+=( '--seed' ) && args+=( $SEED )

echo
echo Configuration:
echo gen_num=$GEN_NUM
echo Logging=$LOG
echo Seed=$SEED

echo && echo -e ${BLUE}lbforaging${NC}  && 
python run_es.py --cfg-path conf/lbf/lbf-8x8-2p-2f-c.yaml "${args[@]}" 
python run_es.py --cfg-path conf/lbf/lbf-10x10-3p-3f.yaml "${args[@]}"
python run_es.py --cfg-path conf/lbf/lbf-15x15-4p-5f.yaml "${args[@]}"
