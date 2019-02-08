# Basic GitHub Instructions
These are just a few basic functions for initial installation and commands I discovered myself using on a daily basis.


## Getting Started
### Initial Installation
You will first need to download your own local repository to start using and modifying. 

- Initial installation
`git install https://github.com/mbogden/galaxyJSPAM.git`


### Change branch.  
There are two branches in the repository.  We would like to keep "master" as an always functional version of the product.  Switch to "Working" branch as you modify and develop files and stay on 'Working'.  I, Matthew, will maintain the updates and merges to 'master' for now.

  - Lists branches 
  `git branch`

  - Change to Working branch 
  `git checkout Working`



## Common/Daily Instructions

### Update your local repository.  
As changes are made to the online repository you will need to update your own local repository to reflect those changes.   It's wise to do this frequently.

- `git pull`


### Making changes.  
As you modifying files in the repository such as adding and modifying files, they are NOT automatically updated in the online repository.  Once you reach checkpoints, you will need to push them to the online repository.  This is done in a couple steps. 

1. Once you've made changes to a file, you will need to add it to the next commit.  You can add multiple files to the same commit.  

  - `git add file.txt`

2.  Commits are saved locally first.  It's good practice to make a 'commit' whenever you create or change a file, even if it's a minor change.  It will save your progress and please always attach a message/comment specifying the change.

  - `git commit -m 'This is a comment.  Ex. Altered function Foo to file.c'`

3. Push your local commit to online repository.  Pushing it out to the online repository will allow everyone to see and benefit from the changes.  It is wise to first always pull from the repository before pushing your own changes or bad things may happen. 

  - `git pull`
  - `git push`


