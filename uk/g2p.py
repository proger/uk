import os
import warnings

warnings.warn('Setting CUDA_VISIBLE_DEVICES="" due to bug in ukro-g2p==0.1.5')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from ukro_g2p.predict import G2P

g2p = G2P('ukro-base-uncased')