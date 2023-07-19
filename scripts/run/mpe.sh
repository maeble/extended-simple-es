BLUE='\033[0;34m'
NC='\033[0m'

GEN_NUM=5

echo && echo -e ${BLUE}MPE Multi-Particle Environments${NC}  && 
python run_es.py --cfg-path conf/mpe/simple_adversary.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/mpe/simple_speaker_listener.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/mpe/simplespread.yaml --generation-num $GEN_NUM
