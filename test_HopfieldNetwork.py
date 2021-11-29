
import numpy as np



def test_hebbian_weights():
  assert(np.allclose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])), np.array([[0., 0.33333333, -0.33333333, -0.33333333], [0.33333333, 0., -1., 0.33333333], [-0.33333333, -1, 0., -0.33333333], [-0.33333333, 0.33333333, -0.33333333, 0.]])))
    
    
def test_storkey_weights():
  assert(np.allclose(storkey_weigths(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])),
  np.array([[1.125, 0.25, -0.25, -0.5],[0.25, 0.625, -1, 0.25],[-0.25, -1, 0.625, -0.25], [-0.5, 0.25, -0.25, 1.125]])))
  

def test_pattern_match():
  assert(np.array([[0,5,3,4], [0,0,0,0]]), np.array([[1,1,2,1], [1,2,3,4]])
 
 
 def test_update():
  assert(update(np.array([[2,5,6,7],[4,5,6,9]]), np.array([[1,1],[1,1]]))-np.array([[1, 1, 1, 1],
          [1, 1, 1, 1]])).all()
 
 
 def test_update_async():
  assert(update_async(np.array([[8,9], [0,0]]), np.array([[1,1],[2,2]])) - np.array([[1, 1]])).all()
  

def test_symmetry_hebbian_weights():
    assert (np.allclose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])), np.transpose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])))))
  

def test_symmetry_storkey_weights():
    assert (np.allclose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])), np.transpose(hebbian_weights(np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])))))
         
         

