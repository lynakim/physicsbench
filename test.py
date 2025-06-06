# %%
era_height = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50]
era_level_height = era_height

print(era_level_height)

for i in range(len(era_height)-2, -1, -1):
    print(i)
    era_level_height[i] += era_height[i+1]
    print(era_level_height)
# %%
