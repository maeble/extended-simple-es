BLUE='\033[0;34m'
NC='\033[0m'

GEN_NUM=5

echo && echo -e ${BLUE}rware${NC}  && 
python run_es.py --cfg-path conf/rware/rware-small-4p.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/rware/rware-tiny-2p.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/rware/rware-tiny-4p.yaml --generation-num $GEN_NUM