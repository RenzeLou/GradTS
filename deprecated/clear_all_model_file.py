'''
 warning: this script used to clear all model checkpoints
'''

import shutil
path = "./checkpoints/"
shutil.rmtree(path,ignore_errors=True)