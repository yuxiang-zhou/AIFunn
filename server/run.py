#!/vol/phoebe/yz4009/src/miniconda3/envs/gitdev/bin/python
from app import app

app.run(debug=True,port=8101,host='0.0.0.0', threaded=True)
