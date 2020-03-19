import hashlib
import sys

import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, 'cognival')

from handlers.plot_handler import *

@pytest.fixture
def history():
    history = {'loss': np.array([0.25488953, 0.21478604, 0.17838786, 0.13976133, 0.1034061 ,
       0.07564405, 0.05683547, 0.04476263, 0.03682742, 0.03132567,
       0.02730239, 0.02424013, 0.02183606, 0.01990202, 0.01834059,
       0.01705953, 0.01599658, 0.01511453, 0.01438425, 0.01375044,
       0.01320592, 0.0127492 , 0.01233769, 0.01198298, 0.01166581,
       0.01139069, 0.01113193, 0.01091016, 0.01070138, 0.01051345,
       0.010342  , 0.01018741, 0.01004625, 0.00990824, 0.00978539,
       0.00967729, 0.00957047, 0.00947993, 0.00939412, 0.00931389,
       0.00923766, 0.00916925, 0.00910349, 0.00904588, 0.00899486,
       0.00894119, 0.0088909 , 0.0088536 , 0.00881352, 0.00876885,
       0.00873459, 0.00870241, 0.00867393, 0.00864445, 0.00861755,
       0.00859891, 0.00857205, 0.00855266, 0.00853562, 0.00851864,
       0.00850827, 0.00848854, 0.00847772, 0.00846338, 0.00845258,
       0.00843623, 0.00842589, 0.00841956, 0.00841243, 0.00840262,
       0.00839491, 0.0083887 , 0.00838005, 0.00837448, 0.0083664 ,
       0.00836016, 0.00835922, 0.00834551, 0.00834227, 0.00833735,
       0.0083334 , 0.00832747, 0.00832495, 0.00832293, 0.00831511,
       0.00830849, 0.00830687, 0.00830639, 0.00829938, 0.00829608,
       0.00829652, 0.00829227, 0.00829248, 0.00828477, 0.00828654,
       0.00828057, 0.00828322, 0.00827282, 0.00827123, 0.00827003]),
       'val_loss': np.array([0.23047497, 0.19465619, 0.15752604, 0.11908325, 0.08707609,
       0.06454188, 0.04988362, 0.04041127, 0.03397305, 0.02937525,
       0.02590852, 0.02320042, 0.02105118, 0.01934954, 0.0179424 ,
       0.01681227, 0.01586363, 0.01507265, 0.01439631, 0.0138424 ,
       0.01334462, 0.01292574, 0.01254491, 0.01221056, 0.01193183,
       0.01166357, 0.01143143, 0.0112074 , 0.01101763, 0.01082873,
       0.01065693, 0.01050958, 0.01037207, 0.01023139, 0.01010608,
       0.00999653, 0.00988527, 0.00979189, 0.00969938, 0.00960981,
       0.00953708, 0.00945923, 0.00938345, 0.00931567, 0.0092655 ,
       0.00920798, 0.00915692, 0.00910952, 0.00905618, 0.00901505,
       0.00898051, 0.00894775, 0.00891546, 0.0088867 , 0.00885898,
       0.00883877, 0.00881933, 0.00879048, 0.00877203, 0.00876519,
       0.00873774, 0.00872758, 0.00871757, 0.00869121, 0.00868277,
       0.00866514, 0.008663  , 0.00864658, 0.00863893, 0.00862224,
       0.00861719, 0.00861065, 0.00859434, 0.0085942 , 0.00859378,
       0.0085781 , 0.00857631, 0.0085627 , 0.00855698, 0.00855979,
       0.00855273, 0.00854975, 0.00853534, 0.00854033, 0.00853054,
       0.0085231 , 0.00852784, 0.0085172 , 0.00851355, 0.00851821,
       0.00851931, 0.00851091, 0.00850721, 0.00850302, 0.00850081,
       0.00850148, 0.00850097, 0.00849433, 0.00848764, 0.00849617])}
    return history

# TODO: Find a better way to tests plots

#def test_plot_handler(tmpdir, history):
#    title = 'test_plotHandler'
#    hasher = hashlib.md5()
#    tmpdir.mkdir('output')
#    plot_handler(title, history, {'wordEmbedding': 'test_embedding'}, str(tmpdir / 'output'))
#    with open(tmpdir / 'output' / 'test_embedding.png', 'rb') as f:
#        buf = f.read()
#        hasher.update(buf)
#        assert hasher.hexdigest() == '36c4d7e67abf8f61032b87944222de9b'
