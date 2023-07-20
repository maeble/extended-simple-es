BLUE='\033[0;34m'
NC='\033[0m'

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GEN_NUM=5

# ----------------------------------------------------------------------------

args=()
[ ! -z "$GEN_NUM" ] && args+=( "--generation-num" ) && args+=( "$GEN_NUM" ) 
[ $LOG == true ] && args+=( '--log' )

echo
echo Configuration:
echo gen_num=$GEN_NUM
echo Logging=$LOG

echo && echo -e ${BLUE}rware${NC}  && 
python run_es.py --cfg-path conf/rware/rware-small-4p.yaml "${args[@]}" 
python run_es.py --cfg-path conf/rware/rware-tiny-2p.yaml "${args[@]}" 
python run_es.py --cfg-path conf/rware/rware-tiny-4p.yaml "${args[@]}" 