import os
from setuptools import setup
from git import Repo, InvalidGitRepositoryError

def get_git_hash():
    try:
        repo = Repo(os.path.dirname(__file__), search_parent_directories=True)
        hash = repo.git.rev_parse(repo.head, short=True)
        if repo.is_dirty():
            hash += '(changed)'
    except InvalidGitRepositoryError:
        hash = "unknown"
    return hash

def write_git_hash():
    hash = get_git_hash()
    with open("GIT_HASH", "w") as f:
        f.write(hash)

write_git_hash()
setup()
