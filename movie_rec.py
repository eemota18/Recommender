import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

def recommendation(model,data,user_ids):

	num_users , num_items = data['train'].shape
	for uid in user_ids:
		pos = data['item_labels'][data['train'].tocsr()[user_ids].indices] 
		
		scores = model.predict(uid, np.arange(num_items))
		top = data['item_labels'][np.argsort(-scores)]

		print("User %s" % uid)
        print("	Known:")
        
        for x in pos[:3]:
            print("        %s" % x)

        print("	Recommended:")
        
        for x in top[:3]:
            print("        %s" % x)
        



data = fetch_movielens(min_rating = 5.0)
model = LightFM(loss='warp')

model.fit(data['train'],epochs=30,num_threads = 2)
recommendation(model, data, [102, 34, 56])
