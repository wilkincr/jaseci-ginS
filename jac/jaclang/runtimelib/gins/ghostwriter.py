import os
import git
from github import Github
from datetime import datetime

class GhostWriter:
    def __init__(self, model, repo_path, github_token, github_repo_fullname):
        self.model = model
        self.repo_path = repo_path
        self.github = Github(github_token)
        self.repo_remote = github_repo_fullname  # e.g., 'your-org/your-repo'

    def rewrite_content(self, suggestions, file_path_relative):
        file_path = os.path.join(self.repo_path, file_path_relative)

        # Step 1: Read original content
        with open(file_path, 'r') as f:
            original_text = f.read()

        # Step 2: Use LLM to fix errors
        response = self.model.fix_errors(suggestions, original_text)
        print("[LLM Response Preview]\n", response[:200])

        # Step 3: Write back to the original file
        with open(file_path, 'w') as f:
            f.write(response)

        # Step 4: Commit and push changes on a new branch
        repo = git.Repo(self.repo_path)
        branch_name = f"llm-fix-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        repo.git.checkout('HEAD', b=branch_name)
        repo.git.add(file_path_relative)
        repo.index.commit(f"LLM fix: updated {file_path_relative}")
        origin = repo.remote(name='origin')
        origin.push(branch_name)

        # Step 5: Open a pull request
        gh_repo = self.github.get_repo(self.repo_remote)
        pr = gh_repo.create_pull(
            title=f"LLM fix suggestions for {file_path_relative}",
            body=f"Automated changes made by LLM to {file_path_relative} based on:\n\n{suggestions}",
            head=branch_name,
            base="main"  # change if your default branch is not "main"
        )

        print(f"✅ PR created: {pr.html_url}")
