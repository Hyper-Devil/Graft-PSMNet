import math 
import matplotlib.pyplot as plt


epoch_total = 10
lr_init = 0.001
epoch=[] 
lr_list=[]
for epoch_now in range(1,epoch_total+1):
    warmup_epoch = math.ceil(epoch_total * 0.1)
    if epoch_now <= warmup_epoch and warmup_epoch == 1:
        lr = lr_init * 0.25 
    elif epoch_now <= warmup_epoch and warmup_epoch > 1:
        lr = lr_init * epoch_now / warmup_epoch
    else:
        lr = lr_init * 0.1 + (lr_init-lr_init * 0.1)*(1 + math.cos(
            math.pi * (epoch_now - warmup_epoch) / (epoch_total - warmup_epoch))) / 2

    epoch.append(epoch_now)
    lr_list.append(lr)

fig = plt.figure()
plt.plot(epoch, lr_list)
# plt.show()
fig.savefig("first.png")
    
