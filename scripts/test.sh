BLUE='\033[0;34m'
NC='\033[0m'


echo && echo -e ${BLUE}mpe${NC}  && 
python run_es.py --cfg-path conf/mpe/simplespread.yaml --generation-num 2 &&  
python run_es.py --cfg-path conf/mpe/simple_speaker_listener.yaml --generation-num 2 && 
python run_es.py --cfg-path conf/mpe/simple_adversary.yaml --generation-num 2 && 

echo && echo -e ${BLUE}lbforaging${NC}  && 
python run_es.py --cfg-path conf/lbf/lbf-8x8-2p-2f-c.yaml --generation-num 2  && 

echo && echo -e ${BLUE}rware${NC}  && 
python run_es.py --cfg-path conf/rware/rware-tiny-2p.yaml --generation-num 2  && 

echo && echo -e ${BLUE}simple-es single-agent configurations${NC}  && 
python run_es.py --cfg-path conf/bipedalwalker.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/cartpole.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/halfcheetah.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/lunarlander_openai.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/lunarlander.yaml --generation-num 2 
