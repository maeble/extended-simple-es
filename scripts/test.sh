BLUE='\033[0;34m'
NC='\033[0m'

echo && echo -e ${BLUE}spread${NC}  && 
python run_es.py --cfg-path conf/simplespread.yaml --generation-num 2 && 

echo && echo -e ${BLUE}speaker listener${NC}  && 
python run_es.py --cfg-path conf/simple_speaker_listener.yaml --generation-num 2 && 

echo && echo -e ${BLUE}adversary${NC}  && 
python run_es.py --cfg-path conf/simple_adversary.yaml --generation-num 2 && 

echo && echo -e ${BLUE}lbforaging${NC}  && 
python run_es.py --cfg-path conf/lbf.yaml --generation-num 2  && 

echo && echo -e ${BLUE}rware${NC}  && 
python run_es.py --cfg-path conf/rware.yaml --generation-num 2  && 

echo && echo -e ${BLUE}simple-es single-agent configurations${NC}  && 
python run_es.py --cfg-path conf/bipedalwalker.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/cartpole.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/halfcheetah.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/lunarlander_openai.yaml --generation-num 2  && 
python run_es.py --cfg-path conf/lunarlander.yaml --generation-num 2 
