#%%
print("Hello world")


# %%
lines.sort()


# %%
# %%


# %%
lines = []
with open("1501.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            lines.append(line)
            print(line)

        lines.sort()
