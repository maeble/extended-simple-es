BLUE='\033[0;34m'
NC='\033[0m'

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

GEN_NUM=20500 # *100 time_steps per generation = 2050000 total timesteps

# ----------------------------------------------------------------------------

args=()
[ ! -z "$GEN_NUM" ] && args+=( "--generation-num" ) && args+=( "$GEN_NUM" ) 
[ $LOG == true ] && args+=( '--log' )

echo
echo Configuration:
echo gen_num=$GEN_NUM
echo Logging=$LOG

echo && echo -e ${BLUE}MPE Multi-Particle Environments${NC}  && 
python run_es.py --cfg-path conf/mpe/simple_speaker_listener.yaml "${args[@]}" 
python run_es.py --cfg-path conf/mpe/simplespread.yaml "${args[@]}" 
