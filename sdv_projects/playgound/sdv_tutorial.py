#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sdv.datasets.demo import download_demo

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests'
)


# In[11]:


real_data.head()


# In[12]:


metadata.visualize()


# In[13]:


from sdv.lite import SingleTablePreset

synthesizer = SingleTablePreset(
    metadata,
    name='FAST_ML'
)


# In[14]:


synthesizer.fit(
    data=real_data
)


# In[15]:


synthetic_data = synthesizer.sample(
    num_rows=500
)

synthetic_data.head()


# In[16]:


sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']

real_data[sensitive_column_names].head(3)


# In[17]:


synthetic_data[sensitive_column_names].head(3)


# In[18]:


from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)


# In[19]:


quality_report.get_visualization('Column Shapes')


# In[20]:


from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='amenities_fee',
    metadata=metadata
)

fig.show()


# In[21]:


from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['checkin_date', 'checkout_date'],
    metadata=metadata
)

fig.show()


# In[22]:


synthesizer.save('my_synthesizer.pkl')

synthesizer = SingleTablePreset.load('my_synthesizer.pkl')


# In[ ]:




