import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def barplot(table, x_col, y_col, hue_col, title, x_label, y_label, size, color):
    """Plot a barplot x-axis: categories
                      y-axis: # of counts
                      
                      
       Args: table
             x_col: column plot on x-axis. 
             y_col: column plot on y-axis(left).
             hue: category column 
             title: str
             x_label:str
             y_label:str
             size: tuple, figure size
             color: list
       Returns:  a bar plot  
    """
    #violin plot
    sns.set_style("white")
    sns.set(font = 'Times New Roman')
    fig, ax = plt.subplots(figsize=size)
    #colors = len(table[y_col])*['#ff7f50']
    ax = sns.barplot(x= x_col, y= y_col, hue = hue_col, data=table, color = color, saturation = 0.6)
    
    #set y ticks range 
    #start, end = ax.get_ylim()
    #ax.yaxis.set_ticks(np.arange(0, end, stepsize))
    #title 
    ax.set_title(title, size = 40)

    #enlarge font-size 
    for tl in ax.get_yticklabels():
            tl.set_fontsize(40)
    #x tick label 
    ax.set_xticklabels(table[x_col].unique())
    for tl in ax.get_xticklabels():
        tl.set_fontsize(40)
     
    #set legend location/size 
    ax.legend(bbox_to_anchor=(1, 1.2), loc=2, prop={'size':40})
    
    #ax.set_title(supplier + ' Demand Forecast vs Score Prediction of NONRFP items Error % in '+time, size = 50)
    #x, y labels 
    ax.set_xlabel(x_label, size=40)
    ax.set_ylabel(y_label, size=40)
    ax.set_axis_bgcolor('white')


def boxplot(table, x_col, y_col, hue_col, title, x_label, y_label):
    """Plot a boxplot x-axis:categories.
                      y-axis:
                      
       Args: table
             time: a string. 
             x_col: column plot on x-axis. 
             y_col: column plot on y-axis
             hue_col: categories. 
             title: str
             x_label: str
             y_label: str
       Returns:  a box plot
    """
    #violin plot
    sns.set_style("white")
    sns.set(font = 'Times New Roman')
    fig, ax = plt.subplots(figsize=(40, 10))
    ax = sns.boxplot(x= x_col, y= y_col, hue = hue_col, data=table, palette= None, saturation = 0.75)
    
    #set y ticks range 
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.1))
    #title 
    ax.set_title(title, size = 45)

    #enlarge font-size 
    for tl in ax.get_yticklabels():
            tl.set_fontsize(30)
    #x tick label 
    ax.set_xticklabels(table[x_col].unique(), rotation = 90)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(40)
        
    #set legend location/size 
    ax.legend(bbox_to_anchor=(1, 1.2), loc=2, prop={'size':30})
    
    #ax.set_title(supplier + ' Demand Forecast vs Score Prediction of NONRFP items Error % in '+time, size = 50)
    #x, y labels 
    ax.set_xlabel(x_label, size=45)
    ax.set_ylabel(y_label, size=45)
    ax.set_axis_bgcolor('white')


#what is the 
def histogram(table, title, x_label, y_label, binwidth):
  """Histogram
     Args: table
           title: str
           x_label: str
           y_label: str
           binwidth: int/float 
    Returns: a histogram
  """
  sns.set_style("white")
  sns.set(font = 'Times New Roman')
  fig, ax = plt.subplots(figsize=(35, 10))
  binrange = np.arange(min(table), max(table) + binwidth, binwidth) 
  #histogram plot 
  sns.distplot(table, kde = False, hist = True, bins = binrange, color = '#32cd32')
  
  #change x-tick fontsize 
  for tl in ax.get_yticklabels():
              tl.set_fontsize(30)
  #x tick label 
  for tl in ax.get_xticklabels():
      tl.set_fontsize(30)

  ax.set_title(title, size = 40)
  ax.set_xlabel(x_label, size=40)
  ax.set_ylabel(y_label, size=40)

def scatterplot(table, x_col, y_col, title, x_label, y_label):
  """Scatterplot: relationship of x and y
     Args: table
           x_col: str
           y_col: str
           title: str
           x_label: str
           y_label: str
    Returns: a scatterplot
  """
  sns.set_style("white")
  sns.set(font = 'Times New Roman')
  fig, ax = plt.subplots(figsize=(18, 4))
  sns.regplot(x= x_col, y= y_col, data= table)
  
  #change x-tick fontsize 
  for tl in ax.get_yticklabels():
      tl.set_fontsize(15)
  #x tick label 
  for tl in ax.get_xticklabels():
      tl.set_fontsize(15)

  ax.set_title(title, size=20)
  ax.set_xlabel(x_label, size=20)
  ax.set_ylabel(y_label, size=20)
  ax.set_axis_bgcolor('white')


def pointplot(table, x_col, y_col, hue_col, title, x_label, y_label, color, size):
    """Pointplot:
       Args: table
             x_col: str
             y_col: str
             hue_col: str
             title: str
             x_label: str
             y_label: str
             color: list 
             size: tuple, figuresize
       Returns: a pointplot
    """
    
    sns.set(font = 'Times New Roman')
    fig, ax = plt.subplots(figsize= size)
    #point plot
    sns.pointplot(x = x_col , y = y_col, hue = hue_col, data = table, palette = color)
    #set title
    ax.set_title(title, size = 40)
    
    #enlarge font-size
    for tl in ax.get_yticklabels():
        tl.set_fontsize(40)
    #x tick label
    #ax.set_xticklabels(table[x_col].unique())
    for tl in ax.get_xticklabels():
        tl.set_fontsize(40)
    #set legend location/size
    ax.legend(bbox_to_anchor=(1, 1.2), loc=2, prop={'size':30})
    #x, y labels
    ax.set_xlabel(x_label , size=40)
    ax.set_ylabel(y_label , size=40)
    
    #set background color and grid
    ax.set_axis_bgcolor('white')
    ax.grid(color='black', linestyle='--', linewidth=2, alpha = 0.5)
    ax.set_ylim(0, 1, 0.2)
