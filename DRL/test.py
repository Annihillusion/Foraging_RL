import os

file_list = os.listdir("exp_records/new_reward")
for file_name in file_list:
    if file_name.startswith("nr"):
        os.rename("exp_records/new_reward/"+file_name, "exp_records/new_reward/"+file_name[3:])

