from api import start_server 
from utils.data_conversion import reformat_json 
from config.constants import Games


if __name__ == '__main__':
    # start_server()
    out = reformat_json("{}", Games.LORCANA)
        
    print(out)

