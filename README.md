# Perspective_Transformation_Layer
Code and experiments for perspective transformation Layer

### How to use perspective transformation layer

```
from pers_layer impoort *
from utils import insert_layer

updated_model = insert_layer(0,4,my_model=model)
```
Note: Here insert layer will insert PT layer(s) at the specified position. 
In the above code, the first argument is the position to insert PT layer in the model; second argument is the number of transformation matrix and third 
argument is the model we want to insert PT layer.

