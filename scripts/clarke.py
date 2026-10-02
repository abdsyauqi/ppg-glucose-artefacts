import numpy as np
def zones(ref, pred):
    z=[]
    for r,p in zip(ref,pred):
        if (r<=70 and p<=70) or (0.8*r<=p<=1.2*r): z.append('A')
        elif (r>=180 and p<=70) or (r<=70 and p>=180): z.append('E')
        elif ((70<=r<=290) and p>=r+110) or ((130<=r<=180) and p<=(7/5)*r-182): z.append('C')
        elif (r>=240 and 70<=p<=180) or (r<=175/3 and 70<=p<=180) or ((175/3<=r<=70) and p>=(6/5)*r): z.append('D')
        else: z.append('B')
    z=np.array(z); return {k:100*(z==k).mean() for k in 'ABCDE'}
