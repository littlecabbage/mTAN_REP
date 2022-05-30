from numpy import full
import pandas as pd


# Loss = []
# roc = []
# with open("/root/zengzihui/ISST/ISST_Baselines/mTAN/ExperimentResult_0428/beta3.txt") as f:
#     lines = f.readlines()

# beta3_Loss = lines[0::3]
# beta3_roc = lines[2::3]


# beta3_Loss = [float(x.split(' = ')[1]) for x in beta3_Loss]
# beta3_roc = [float(x) for x in beta3_roc]


# with open("/root/zengzihui/ISST/ISST_Baselines/mTAN/ExperimentResult_0428/beta5.txt") as f:
#     lines = f.readlines()

# beta5_Loss = lines[0::3]
# beta5_roc = lines[2::3]


# beta5_Loss = [float(x.split(' = ')[1]) for x in beta5_Loss]
# beta5_roc = [float(x) for x in beta5_roc]


# # print(beta3_roc)
# df = pd.DataFrame([beta3_Loss, beta3_roc, beta5_Loss, beta5_roc])

# df.T.to_csv("/root/zengzihui/ISST/ISST_Baselines/mTAN/ExperimentResult_0428/result.csv", index= None)


import pandas as pd
with open("/root/zengzihui/ISST/ISST_Baselines/mTAN/ExperimentResult_0428/full.txt") as f:
    lines = f.readlines()

full_Loss = lines[0::3]
full_roc = lines[2::3]

full_Loss = [float(x.split(' = ')[1]) for x in full_Loss]
full_roc = [float(x) for x in full_roc]

print(full_roc)
df = pd.DataFrame([full_Loss, full_roc])
df.T.to_csv("/root/zengzihui/ISST/ISST_Baselines/mTAN/ExperimentResult_0428/full_result.csv", index= None)