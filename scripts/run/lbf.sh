BLUE='\033[0;34m'
NC='\033[0m'

GEN_NUM=5

echo && echo -e ${BLUE}lbforaging${NC}  && 
python run_es.py --cfg-path conf/lbf/lbf-8x8-2p-2f-2s-c.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-8x8-2p-2f-c.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-10x10-3p-3f-2s.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-10x10-3p-3f.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-15x15-3p-5f.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-15x15-4p-3f.yaml --generation-num $GEN_NUM
python run_es.py --cfg-path conf/lbf/lbf-15x15-4p-5f.yaml --generation-num $GEN_NUM
