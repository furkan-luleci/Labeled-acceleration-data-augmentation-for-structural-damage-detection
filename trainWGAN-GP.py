import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import pandas as pd
from numpy import std
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from utils import gradient_penalty

# Hyperparameters and others
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.00001
BATCH_SIZE = 1024
CHANNELS_IMG = 1
Z_DIM = 1024
NUM_EPOCHS = 600
FEATURES_CRITIC = 1024
FEATURES_GEN = 1024
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10

# Clear cache
torch.cuda.empty_cache()

## Data loading
a11=pd.read_excel('repository', usecols='A')

a11 = pd.DataFrame(a11)

# To tensor
a11=torch.tensor(a11.values)

# Vector extracting
a11=torch.squeeze(a11, 0)
a11=torch.unsqueeze(a11, 1)

loader = torch.utils.data.DataLoader(a11, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, amsgrad=True)
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, amsgrad=True)

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1).to(device)
writer = SummaryWriter()

step = 0
generateddata = []
realdata = []

# Define FID score
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), std(act1)
	mu2, sigma2 = act2.mean(axis=0), std(act2)
	# calculate sum squared difference between means
	diff = (mu1 - mu2)**2.0
	# calculate sqrt of product between cov
	stdmean = sigma1*sigma2
	# calculate score
	fid = diff + sigma1**2 + sigma2**2 - 2.0 * stdmean
	return fid

scaler=MinMaxScaler(feature_range=(-1,1))
noisen=(scaler.fit_transform(np.random.default_rng().
            normal(0 ,1, 1024).reshape(-1,1)))  # Add decaying noise over time
noisen=torch.squeeze(torch.tensor(noisen))
noisen=torch.unsqueeze(noisen,1)
noisen=torch.unsqueeze(noisen,1)

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! unsupervised
    for batch_idx, (real) in enumerate(loader):
        # adding noise
        real = real.to(device) + torch.pow(noisen,(step+1)).to(device=device, dtype=torch.float)
        # fixing some dimensions, there is a better way to do this with "torch.swapaxes"
        cur_batch_size = real.shape[0]
        real=torch.squeeze(real, 1)
        real=torch.squeeze(real, 0)
        real=torch.unsqueeze(real, 0)
        real=(torch.unsqueeze(real, 0)).to(device=device, dtype=torch.float)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            
            # clip critic weights between -0.01, 0.01
            # for p in critic.parameters():
            #    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                data3 = real.cpu()
                data3 = np.array(data3)
                data3 = data3[0]
                data3 = np.squeeze(data3)
                fake3 = fake.cpu()
                fake3 = np.array(fake3)
                fake3 = fake3[0]
                fake3 = np.squeeze(fake3)
                fidscore = calculate_fid(data3,fake3)
                cosdistance = distance.euclidean(data3,fake3)
                generateddata.append(fake3)
                realdata.append(data3)
                writer.add_scalar("fidscore", fidscore, global_step=step)
                writer.add_scalar("loss_critic", loss_critic, global_step=step)
                writer.add_scalar("loss_gen", loss_gen, global_step=step)
                writer.add_scalar("cosdistance", cosdistance, global_step=step)

            step += 1
            gen.train()
            critic.train()
            
        torch.save(critic, 'gen.pkl')
        torch.save(gen, 'critic.pkl')


# Generate data            
gen.eval()
critic.eval()
generateddata_final = []
realfinal_data = []
fidlist = []

with torch.no_grad():
    for _ in range(256):
        fakefinal = gen(torch.randn(1, Z_DIM, 1).to(device))
        fakefinal = (fakefinal.cpu()).detach()
        fakefinal = np.array(fakefinal)
        fakefinal = fakefinal[0]
        fakefinal = fakefinal[0]
        generateddata_final.append(fakefinal)
generateddata_final = np.transpose(np.asarray(generateddata_final))

with torch.no_grad():
    for batch_idx, (real) in enumerate(loader):
        real = real.to(device)
        realfinal = real.cpu()
        realfinal = np.array(realfinal)
        realfinal = np.squeeze(realfinal)
        realfinal_data.append(realfinal)
realfinal_data = np.transpose(np.asarray(realfinal_data))

# Calculate the FID score in each column in generateddata_final and realfinal_data
for i in range(256):
    colr = generateddata_final[:,i]
    for ix in range(256):
        colf = realfinal_data[:,ix]
        fidlist.append(calculate_fid(colr,colf))
fidlist = np.array(fidlist)
