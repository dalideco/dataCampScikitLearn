#in case of 0 or 1 data we use seaborn's countplot
plt.figure() #initialize the figure 
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
#set the variable in x axis and hue as party takes the data and color palette that satisfies color_palette()
plt.xticks([0,1], ['No', 'Yes'])#shows the 0  in the x axis as No and the 1 as Yes
plt.show()#shows the figure

#rq: the y axis is count by default