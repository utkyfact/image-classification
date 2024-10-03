
# # Image Classification using Transfer Learning

import numpy as np
from PIL import Image
from IPython.display import Image as show_image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions


# In[22]:


img = Image.open("sportscar.jpg").resize((299,299))


# In[23]:


img = np.array(img)


# In[24]:


img.shape


# In[25]:


print(img.ndim)


# In[26]:


img = img.reshape(-1,299,299,3)
# In[27]:


img.shape


# In[28]:


print(img.ndim)


# In[7]:


img = preprocess_input(img)   



# In[8]:


incresv2_model = InceptionResNetV2(weights='imagenet', classes=1000)


# In[9]:


print(incresv2_model.summary())
print(type(incresv2_model))


# In[10]:


show_image(filename='sportscar.jpg') 


# In[11]:


preds = incresv2_model.predict(img)
print('Predicted categories:', decode_predictions(preds, top=2)[0])



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### reshape

# In[13]:


import numpy as np

arr = np.arange(24)

print(arr)


# In[14]:


arr.shape


# In[15]:


arr.ndim


# In[16]:


arr = arr.reshape(-1,24)


# In[17]:


arr.shape


# In[18]:


arr


# In[19]:


arr.ndim


# In[ ]:




