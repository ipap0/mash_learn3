import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

learn_data = [-10, -2, 0, 5, 15, 22, 23, 23, 24, 27, 29, 30, 45, 55, 61, 70]
learn_df = pd.DataFrame({'nums':learn_data})

test_data = [1, 15, 25, 30]

scaler = StandardScaler()

scaled_learn_data = scaler.fit_transform(learn_df)
print(scaled_learn_data)


# learn_df['num_bins']= pd.cut(learn_df['nums'], bins=7)
# print(learn_df['num_bins'].value_counts())
# learn_df['num_bins'].value_counts().plot.bar()
#
# plt.show()

work_data = [20, 45, 295]
work_df = pd.DataFrame(work_data)

scaled_work_data = scaler.transform(work_df)
print(scaled_work_data)

