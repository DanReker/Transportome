#####################
# IMPORTS
#####################
import numpy as np
import math
import scipy
from scipy import stats
from joblib import Parallel, delayed
# rdkit for chemistry processing
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
# machine learning models
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.cluster import KMeans
# machine learning metrics
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.metrics import fbeta_score, make_scorer
# machine learning utility functions
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

#####################
# FUNCTIONS TO READ DATA
#####################
def transclass(x):
	if "True" in x:
		return True
	elif "False" in x:
		return False
	else:
		return None


def transid(x):
	try:
		return int(x)
	except:
		return None

def read_mols(filename, idrow=False):
	mols = []
	clas = []
	nams = []
	smis = []
	ids = []
	
	infile = open(filename,"rb")
	infile.next()
	
	for line in infile:
		nam = line.split("\t")[0]
		mol = Chem.MolFromSmiles(line.split("\t")[1])
		cla = [transclass(x.strip()) for x in line.split("\t")[2:5]]
		smi = line.split("\t")[1]
		if mol:
			clas += [cla]
			mols += [mol]
			nams += [nam]
			smis += [smi]
			if idrow:
				ids += [transid(line.split("\t")[-1].strip())]
		
		else:
			print "error " + line.split("\t")[3] + " " + line.split("\t")[4]
			None
		
	
	return [nams, mols, clas, smis, ids]


descr = Descriptors._descList
calc = [x[1] for x in descr]

def describe_mol(mol):
	fp = AllChem.GetMorganFingerprintAsBitVect(mol,4,nBits=2048)
	fp_list = []
	fp_list.extend(fp.ToBitString())
	fp_expl = []
	fp_expl = [float(x) for x in fp_list]
	ds_n = []
	for d in calc:
		v = d(mol)
		if v > np.finfo(np.float32).max: 	# postprocess descriptors for freak large values
			ds_n.append(np.finfo(np.float32).max)
		elif math.isnan(v):
			ds_n.append(np.float32(0.0))
		else:
			ds_n.append(np.float32(v))
	
	return fp_expl + list(ds_n)

def describe_mols(mols):
	mol_descrs = []
	for mol in mols:
		mol_descrs += [describe_mol(mol)];
	
	return mol_descrs


########
# function to standardize calculations of benchmark statistics
def evaluate(pred_vals,true_vals):
	tn, fp, fn, tp = confusion_matrix(true_vals, pred_vals).ravel()
	return [mcc(true_vals,pred_vals),
	accuracy(true_vals,pred_vals),
	bac(true_vals,pred_vals),
	tn, fp, fn, tp]
	


#####################
# INITIAL MODEL EVALUATION
#####################

########
# read training database
train = read_mols("database.tsv")
n = np.array(train[0])
s = np.array(train[3])
x = np.array(describe_mols(train[1]))
y = np.array(train[2])
m = np.array(train[1])

models = [ RandomForestClassifier(n_estimators=500), GaussianNB(),KNeighborsClassifier(3), DecisionTreeClassifier(),MLPClassifier(),LinearSVC(),ExtraTreesClassifier(), GaussianProcessClassifier()]
model_comparision=[]
for model in models:
	
	for i in range(3):
		mask = np.array([a is not None for a in y[:,i]])
		y_sub = y[mask,i].astype(int)
		x_sub = x[mask]
		
		temp = []
		for ii in range(5):
			preds = sklearn.model_selection.cross_val_predict(model,x_sub,y_sub,cv=10,n_jobs=-1)
			temp += [evaluate(preds,y_sub)]
	
		model_comparision+= [temp]


model_comparision = np.array(model_comparision)

for i in range(len(model_comparision)):
	print "\t".join([str(a) + "±" + str(b) for a,b in zip(np.round(np.mean(model_comparision[i,:],axis=0),3),np.round(np.std(model_comparision[i,:],axis=0),3))])




########
# leave cluster out validation based on MACCS similarity clusters
maccs_k = [MACCSkeys.GenMACCSKeys(mol) for mol in m]
kmeans = KMeans(n_clusters=50, random_state=0).fit(maccs_k)
maccs_cluster = kmeans.predict(maccs_k)
cluster_performance = []
for i in range(3):
	temp = []
	
	mask = np.array([a is not None for a in y[:,i]])
	y_sub = y[mask,i].astype(int)
	x_sub = x[mask]
	c_sub = maccs_cluster[mask]
	
	for ii in range(5):
		preds = cross_val_predict(model,x_sub,y_sub,groups=c_sub,cv=LeaveOneGroupOut(),n_jobs=-1)
		temp += [evaluate(preds,y_sub)]
	
	cluster_performance += [temp]

cluster_performance = np.array(cluster_performance)

for i in range(len(cluster_performance)):
	print "\t".join([str(a) + "±" + str(b) for a,b in zip(np.round(np.mean(cluster_performance[i,:],axis=0),3),np.round(np.std(cluster_performance[i,:],axis=0),3))])


########
# predict training compounds for active learning-based data augmentation
preds1 = []
names1 = []
trans1 = []
conf1  = []
smis1  = []
transporters = ['mdr1', 'bcrp', 'mrp2']
for i in range(3):
	mask = np.array([a is not None for a in y[:,i]])
	y_sub = y[mask,i].astype(int)
	x_sub = x[mask]
	model.fit(x_sub,y_sub)
	
	mask2 = np.array([a is None for a in y[:,i]])
	preds1 += model.predict(x[mask2]).tolist()
	names1 += n[mask2].tolist()
	smis1  += s[mask2].tolist()
	trans1 += [transporters[i] for j in range(np.sum(mask2))]
	conf1  += model.predict_proba(x[mask2])[:,1].tolist()


outfile = open("predictions_for_training","wb")
for i in range(len(preds1)):
	outfile.write("\t".join([trans1[i], names1[i], smis1[i], str(preds1[i]), str(conf1[i])]) + "\n")

outfile.flush()
outfile.close()


#########
# evaluate performance on external data

highconf_true = 0
highconf_total= 0

lowconf_true = 0
lowconf_total= 0

for transp in ["mdr1","bcrp","mrp2"]:
	infile = open("additional_" + transp + ".tsv")
	_ = infile.next()
	
	scores = []
	clas = []
	for line in infile:
		scores += [float(line.split("\t")[2])]
		clas +=   [line.split("\t")[3].strip()]
	
	
	for i in range(len(scores)):
		if scores[i] > 0.6:
			highconf_total += 1
			if clas[i] == "substrate":
				highconf_true += 1
		elif scores[i] > 0.5:
			lowconf_total += 1
			if clas[i] == "substrate":
				lowconf_true += 1
		elif scores[i] > 0.4:
			lowconf_total += 1
			if clas[i] == "non substrate":
				lowconf_true += 1
		elif scores[i] < 0.4:
			highconf_total += 1
			if clas[i] == "non substrate":
				highconf_true += 1



print highconf_true * 1.0 / highconf_total
# 0.838709677419
print lowconf_true * 1.0 / lowconf_total
# 0.521739130435



#########
### add literature curated data from active learning search
y2 = y.copy()
transporters = ['mdr1', 'bcrp', 'mrp2']
num_annotations = [49, 34, 20]
for i in range(3):
	cnt = 0
	infile = open("additional_" + transporters[i] + ".tsv","r")
	_ = infile.next()
	for line in infile:
		name = line.split("\t")[0]
		smi = line.split("\t")[1]
		cla = (line.split("\t")[3].strip() == "substrate")
		
		for j in range(len(n)):
			if n[j] == name and s[j] == smi and y[j][i] == None:
				y2[j][i] = cla
				cnt += 1
				break
				
	assert cnt == num_annotations[i]



########
# semi-supervised matrix completion

model = RandomForestClassifier(n_estimators=500)

y_augm = np.array([[None,None,None] for i in range(len(y2))])

for i in range(3):
	mask = np.array([a is not None for a in y2[:,i]])
	mask2 = np.invert(mask)
	
	y_sub = y2[mask,i].astype(int)
	x_sub = x[mask]
	
	model.fit(x_sub,y_sub)
	
	y_augm[mask,i] = y_sub
 	y_augm[mask2,i] = model.predict(x[mask2])
	assert all(a is not None for a in y_augm[:,i])


types = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0]]
for t in types:
	print str(t) + "\t" + str(np.sum([all(a == t) for a in y_augm]))


n_drugbank = np.loadtxt("database_drugbank5_only.tsv",delimiter="\t",usecols=(0,),dtype=object)
drugbank_mask = np.array([name in n_drugbank for name in n])
drugbank_augm = y_augm[drugbank_mask]

sum(np.sum(drugbank_augm,axis=1) > 1) *1.0 / len(drugbank_augm)

types = [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0]]
for t in types:
	print str(t) + "\t" + str(np.sum([all(a == t) for a in drugbank_augm]))


########
# predict model drug library


model = RandomForestClassifier(n_estimators=500)

infile = open("drug_library.tsv")
_ = infile.next()
model_drugs_smiles = []
model_drugs_profiles = []
for line in infile:
	model_drugs_smiles += [line.split("\t")[1]]
	model_drugs_profiles += [[transclass(c.strip()) for c in line.split("\t")[2:5]]]


model_drugs_x = describe_mols([Chem.MolFromSmiles(mol) for mol in model_drugs_smiles])

model_drugs_preds = []
for i in range(3):
	mask = np.array([a is not None for a in y2[:,i]])
	
	y_sub = y2[mask,i].astype(int)
	x_sub = x[mask]
	
	model.fit(x_sub,y_sub)
	model_drugs_preds += [model.predict(model_drugs_x)]


model_drugs_preds = np.transpose(model_drugs_preds)
correct_preds = 0
for pred, prof in zip(model_drugs_preds,model_drugs_profiles):
	correct_preds += np.sum(pred == prof)

print "correct preds: " + str(correct_preds) + " ( " + str(float(correct_preds) / (3*28)) + " % )"




########
# add model library

drug_library = read_mols("drug_library.tsv",True)
y3 = y2.copy()
n3 = n.copy()
x3 = x.copy()
m3 = m.copy()


for i in range(len(drug_library[0])):
	if drug_library[-1][i]:
		y3[drug_library[-1][i]] = np.array(drug_library[2][i])
	else:
		y3 = np.vstack( (y3, np.array(drug_library[2][i]) ) )
		n3 = np.append(n3, drug_library[0][i])
		x3 = np.vstack( (x3, describe_mol(drug_library[1][i]) ) )
		m3 = np.append(m3, drug_library[1][i])


########
## predict investigational compounds

names_investigational = []
id_investigational = []
smiles_investigational = []
molecules_investigational = []
x_investigational = []
profile_investigational = []

infile = open("investigational_compounds.tsv","r")
_ = infile.next()
for line in infile:
	temp = line.split("\t")
	if not Chem.MolFromSmiles(temp[1]) is None:
		id_investigational += [temp[0]]
		smiles_investigational += [temp[1]]
		names_investigational += [temp[2]]
		molecules_investigational += [Chem.MolFromSmiles(temp[1])]
		x_investigational += [describe_mol(Chem.MolFromSmiles(temp[1]))]


assert len(x_investigational) == len(names_investigational)



prediction_confidences = []
for i in range(3):
	mask = np.array([a is not None for a in y2[:,i]])
	y_sub = y2[mask,i]
	x_sub = x[mask]
	
	model.fit(x_sub,y_sub.astype(int))
	
	prediction_confidences += [model.predict_proba(x_investigational)[:,1].tolist()]



conf_matrix = np.transpose(prediction_confidences)
names_investigational = np.array(names_investigational)

for ideal in [[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[0,0,0]]:
	outfile = open("top15_" + str(ideal) + ".txt","w")
	for name in names_investigational[np.argsort([np.sum(np.abs(c-ideal)) for c in conf_matrix])][:15]:
		outfile.write(name +"\n")
    
	outfile.flush()
	outfile.close()




########
# add investigational library after experimental testing


investigational_library = read_mols("investigational_library.tsv",True)
y4 = y3.copy()
n4 = n3.copy()
x4 = x3.copy()
m4 = m3.copy()


for i in range(len(investigational_library[0])):
	if investigational_library[-1][i]:
		y4[investigational_library[-1][i]] = np.array(investigational_library[2][i])
	else:
		y4 = np.vstack( (y4, np.array(investigational_library[2][i]) ) )
		n4 = np.append(n4, investigational_library[0][i])
		x4 = np.vstack( (x4, describe_mol(investigational_library[1][i]) ) )
		m4 = np.append(m4, investigational_library[1][i])




####
# print sizes of datasets
print "\t\tPgp\tBCRP\tMRP2"
for (xi,yi) in [(x,y),(x,y2),(x3,y3),(x4,y4)]:
	print ("Substrates\t"+"\t".join([str(_n) for _n in np.sum(yi==True,axis=0)]))
	print ("Non-Substrates\t"+"\t".join([str(_n) for _n in np.sum(yi==False,axis=0)]))
		


########
# compare models based on cross validation performance

model = RandomForestClassifier(n_estimators=500,oob_score=True)
cross_perfs = []
for (xi,yi) in [(x,y),(x,y2),(x3,y3),(x4,y4)]:
	preds = np.array([])
	vals = np.array([])
	for i in range(3):
		mask = np.array([a is not None for a in yi[:,i]])
		
		y_sub = yi[mask,i].astype(int)
		x_sub = xi[mask]
		
		temp = []
		for ii in range(5):
			preds = cross_val_predict(model,x_sub,y_sub,n_jobs=-1)
			temp += [evaluate(preds,y_sub)]
		
		cross_perfs += [temp]

cross_perfs = np.array(cross_perfs)

for i in range(12):
	print "\t".join([str(a) + "±" + str(b) for a,b in zip(np.round(np.mean(cross_perfs[i,:],axis=0),3),np.round(np.std(cross_perfs[i,:],axis=0),3))])


for i in range(1,4):
	stats.ttest_ind(cross_perfs[2+(3*(i-1)),:,0],cross_perfs[2+3*i,:,0])

########
# compare models based on oob performance

model = RandomForestClassifier(n_estimators=500,oob_score=True)
oob_perfs = []
for (xi,yi) in [(x,y),(x,y2),(x3,y3),(x4,y4)]:
	t_x = np.array([])
	t_y = np.array([])
	
	for i in range(3):
		temp = []
		for ii in range(5):
			mask = np.array([a is not None for a in yi[:,i]])
			
			y_sub = yi[mask,i].astype(int)
			x_sub = xi[mask]
			
			
			_ = model.fit(x_sub,y_sub)
			preds = np.argmax(model.oob_decision_function_,axis=1)
			#preds_proba = model.oob_decision_function_[:,1]
			temp += [evaluate(preds,y_sub)]
	
		oob_perfs += [temp]	



oob_perfs = np.array(oob_perfs)

for i in range(12):
	print "\t".join([str(a) + "±" + str(b) for a,b in zip(np.round(np.mean(oob_perfs[i,:],axis=0),3),np.round(np.std(oob_perfs[i,:],axis=0),3))])



########
# compare models based on oob performance and high confidence predictions

model = RandomForestClassifier(n_estimators=50,oob_score=True)
for (xi,yi) in [(x,y),(x,y2),(x3,y3),(x4,y4)]:
	t_x = np.array([])
	for i in range(3):
		mask = np.array([a is not None for a in yi[:,i]])
		
		y_sub = yi[mask,i].astype(int)
		x_sub = xi[mask]
		
		_ = model.fit(x_sub,y_sub)
		t_x = np.append(t_x,model.oob_decision_function_[:,1])
	
	
	float(np.sum(t_x > 0.8) + np.sum(t_x < 0.2))




########
# compare models based on applicability domain expansion for approved drugs
approved = describe_mols([Chem.MolFromSmiles(s) for s in np.loadtxt("drugbank5_approved.smiles",usecols=(1,),comments=None,dtype=object) if not Chem.MolFromSmiles(s) is None])

approved = np.array(approved)

probas = []
for (xi,yi) in [(x,y),(x,y2),(x3,y3),(x4,y4)]:
	temp = []
	for i in range(3):
		
		mask = np.array([a is not None for a in yi[:,i]])
		
		y_sub = yi[mask,i].astype(int)
		x_sub = xi[mask]
		
		_ = model.fit(x_sub,y_sub)
		temp += model.predict_proba(approved)[:,1].tolist()
	
	probas += [temp]

probas = np.array(probas)

for p in probas:
	print str(float(np.sum(p<0.2) + np.sum(p>0.8)))


