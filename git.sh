#!/bin/bash

git update-index --skip-worktree git.sh

git add .
git commit -m "add"
git push