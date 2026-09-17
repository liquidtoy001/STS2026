"""English, macOS / GitHub website-first content for the STS2026 handbook."""


def populate(page, p, h, step, code, note, table, diagram, branch_diagram):
    page('GitHub team handbook', 'ENGLISH / macOS + WEBSITE FIRST / VERSION 2.0 / 17 SEPTEMBER 2026')
    p('STS2026<br/>One team.<br/>One shared project.', 'cover')
    p('Start in Safari or Chrome on your Mac. Accept your invitation, create a branch, edit a file and open a pull request on GitHub. Use a local development environment when you need to run code.')
    diagram(['Branch', 'Edit + commit', 'Pull request', 'Merge'], ['Your task', 'On GitHub', 'Team review', 'Into main'])
    h('Your project')
    p('<b>Repository:</b> <a href="https://github.com/liquidtoy001/STS2026" color="#007F82">github.com/liquidtoy001/STS2026</a><br/><b>Default teammate platform:</b> macOS<br/><b>Main interface:</b> github.com in your browser<br/><b>Optional local folder:</b> ~/Documents/STS2026<br/><b>Shared baseline branch:</b> main')
    note('For the in-person onboarding session', 'Read page 5 to accept the invitation. Then complete the browser-only practice on page 19. You do not need to install Git or GitHub Desktop for those steps.')
    p('The initial main branch must exist before branch-based exercises can begin. The maintainer setup is on page 4. Examples name proposed files and branches; they do not imply those files already exist.', 'small')
    p('All diagrams are explanatory illustrations. Button names and positions may vary slightly as GitHub changes its interface.', 'small')

    page('Find the right page', 'Joining the team? Start on page 5. Keep page 22 open during everyday work.')
    entries = [
        ('Understand website edits and local Git', 3),
        ('Maintainer: publish the initial baseline', 4),
        ('Accept your invitation and open STS2026', 5),
        ('Create a task branch on the website', 6),
        ('Create, edit or upload files on GitHub', 7),
        ('Open and review a pull request', 8),
        ('Merge and start the next task', 9),
        ('Bring main updates into your PR branch', 10),
        ('Move between website edits and local Mac work', 11),
        ('Work with notebooks and record experiments', 12),
        ('Decide which project files belong in Git', 13),
        ('Fix access, upload and push problems', 14),
        ('Resolve a conflict on the website', 15),
        ('Recover work on the wrong branch', 16),
        ('Undo an edit or recover a deleted file', 17),
        ('Handle large files and exposed secrets', 18),
        ('Complete your first browser-only practice PR', 19),
        ('Maintainer: invitations, rules and reviews', 20),
        ('Official references and glossary', 21),
        ('One-page website workflow checklist', 22),
        ('Optional: set up local development on macOS', 23),
    ]
    for label, number in entries:
        p(f'<a href="#p{number}" color="#007F82"><b>{number:02d}</b> &nbsp; {label}</a>')

    page('Website or local Git?', 'Choose the interface that fits the work; both use the same repository history.')
    table(['On github.com', 'In a local clone on your Mac'], [
        ('Edit a text file or upload a small file, then click Commit changes. The commit is already on GitHub.', 'Edit files in an editor. Commit records them locally; push uploads the commits to GitHub.'),
        ('Select a branch to see its current server-side files. No local pull is needed to view that branch on the website.', 'Fetch/pull to bring remote changes into your local copy. Refreshing a browser tab does not update local files.'),
        ('Create PRs, review changes and discuss issues. Suitable for documentation and small edits.', 'Run notebooks, tests and training in an appropriate environment. Better for substantive code changes.'),
    ])
    diagram(['Web edit', 'Web commit', 'PR review'], ['Browser editor', 'Already on GitHub', 'Not yet in main'])
    table(['Term', 'Meaning'], [
        ('Branch', 'A named line of work. main holds reviewed work; a task branch holds your proposed changes.'),
        ('Commit', 'A recorded snapshot with an author and message. Where you create it determines whether a separate push is needed.'),
        ('Pull request / PR', 'A request to merge your source (compare/head) branch into the target (base) branch.'),
        ('Clone / origin', 'Clone creates a local repository with history. origin is the usual nickname for its GitHub remote.'),
        ('Download ZIP', 'A snapshot of files, without Git history or a configured connection for pushing changes.'),
    ], [110, 393])
    note('Browser editing does not run your code', 'Viewing a notebook, saving an edit or merging a PR is not execution. github.dev is also an editor, not a Python runtime. Use the documented local or team compute environment for verification.')

    page('Publish the first baseline', 'MAINTAINER ONLY / Complete this once before teammates create task branches.')
    p('The local setup previously had no commits. If main is already published when you read this, skip initial creation and follow normal PR workflow. Do not initialize a second, unrelated history in the existing local project.')
    step(1, 'Review the source notebook and intended materials. Exclude credentials, datasets, checkpoints, temporary output and private information. Prepare a README with real setup and data-access instructions.')
    step(2, 'For a small new repository, use the empty repository page link to <b>upload an existing file</b>. Upload only reviewed baseline files and write an initial commit message. If GitHub requires a first file, create README.md first, then use <b>Add file &gt; Upload files</b>.')
    step(3, 'Confirm the initial branch is <b>main</b>, make it the default if needed, and check the uploaded file paths. This initial main commit is the one-time setup exception.')
    step(4, 'Invite teammates using <b>Settings &gt; Collaborators &gt; Add people</b>. Confirm the matching username before sending each invitation. Page 5 explains their acceptance steps.')
    step(5, 'Tell everyone when the baseline is ready. After this, normal changes go through task branches and PRs.')
    note('If an existing local repository is the source', 'Publishing its reviewed first commit with Git is also valid. Choose one initial publication route. If you publish through the website first, clone the published repository into a new folder before integrating local work; do not force-push an independently initialized history.', True)
    h('Check before inviting the whole team')
    p('Open the repository as the maintainer and confirm the baseline files are present, the README explains how to run the project, and no secrets or bulk artifacts were included. Web uploads have a 25 MiB per-file limit; inspect even smaller files before sharing.')

    page('Accept your invitation', 'IN-PERSON ONBOARDING / All steps here happen in your browser.')
    step(1, '<b>Send your GitHub username or profile link.</b> On your Mac, sign in at github.com. Open your avatar menu and choose <b>Your profile</b>. Share the username in github.com/YOUR_USERNAME, not only your display name.')
    step(2, '<b>Wait while the maintainer invites you.</b> Stay signed in to that same account. If you have multiple accounts, check which one is active before opening the invitation.')
    step(3, '<b>Open the invitation email.</b> Check the inbox associated with your GitHub account, including spam/junk. Follow the GitHub invitation link for <b>liquidtoy001/STS2026</b>.')
    step(4, '<b>Click Accept invitation on GitHub.</b> Sign in if prompted and check the owner and repository name. Wording may vary. Opening the email alone does not accept the invitation.')
    step(5, '<b>Confirm with the maintainer.</b> Open the repository link below and say that you accepted. The maintainer should verify you are a collaborator rather than a pending invite. Viewing a public repo alone does not prove write access.')
    p('<a href="https://github.com/liquidtoy001/STS2026" color="#007F82"><b>Open STS2026</b> - github.com/liquidtoy001/STS2026</a>')
    diagram(['Share username', 'Accept invite', 'Open repo'], ['Your profile', 'Email link', 'Same GitHub account'])
    note('No email, invalid link or 404?', 'Check the username, current login and spam folder. Check for a GitHub invitation notification. Ask the maintainer to inspect the pending invite and share its invitation link if available, or resend it.')
    h('Next: your first website task')
    p('Once main contains the initial files, follow page 6 to create a branch, or page 19 for a complete practice PR. Installing GitHub Desktop and cloning the repo are optional until you need local code execution.')

    page('Create a task branch', 'WEBSITE / One short-lived branch for one coherent task.')
    branch_diagram()
    step(1, 'Agree the task and owner. Our running example is adding <b>docs/experiment-guide.md</b>. Coordinate before two people edit the same notebook.')
    step(2, 'Open STS2026 on GitHub and select the <b>Code</b> tab. In the branch dropdown above the files or in the file tree, choose <b>main</b>.')
    step(3, 'Open the dropdown again. Type <b>docs/experiment-guide</b> in the branch search/create field, then choose the option to create it from <b>main</b>.')
    step(4, 'If the shortcut is not shown, choose <b>View all branches &gt; New branch</b>, enter the name, set <b>Source: main</b>, and create the branch.')
    step(5, 'Before editing, confirm the branch selector says <b>docs/experiment-guide</b>. This branch already exists on GitHub; you do not need to publish or push it separately.')
    table(['Example branch', 'Task'], [('docs/experiment-guide', 'Explain a reproducible experiment workflow.'), ('feat/evaluation-summary', 'Add one evaluation feature.'), ('fix/audio-loading', 'Fix a specific input-loading problem.')])
    note('No create-branch option?', 'Confirm your invitation was accepted and you have write access. Do not create an unexpected fork to work around a missing team permission; ask the maintainer to check.')

    page('Edit, create or upload', 'WEBSITE / Always check the task branch before clicking Commit changes.')
    h('Edit an existing text file')
    step(1, 'Select your task branch, open the file and click the pencil / <b>Edit this file</b> control. Make a focused change. Use the preview/diff view if available.')
    step(2, 'Click <b>Commit changes...</b>, write a specific message, and confirm the target. Commit directly to <b>your task branch</b>, not main. Complete the dialog, then inspect the saved file and commit.')
    h('Create a new document')
    step(3, 'On your task branch choose <b>Add file &gt; Create new file</b>. Enter <b>docs/experiment-guide.md</b>; slashes create folders. Add text, preview the Markdown, and commit to the task branch.')
    h('Upload a small file from your Mac')
    step(4, 'Navigate to the intended folder on your task branch. Choose <b>Add file &gt; Upload files</b>. Select the file from Finder, check its destination and filename, then commit. Review the resulting diff, especially if replacing a file.')
    note('A website commit is already uploaded', 'For the normal github.com file editor and upload page, completing Commit changes writes the commit on GitHub. There is no separate Push button. You still need a PR to merge your task branch into main.')
    table(['Good commit message', 'Too vague'], [('docs: explain experiment setup', 'update'), ('fix: handle empty audio input', 'fix stuff')])
    p('Browser uploads allow up to 25 MiB per file and up to 100 files per upload. These are limits, not a recommendation to upload datasets. Review notebook outputs and never upload .env, .git, environments or model weights.', 'small')

    page('Open and review a PR', 'WEBSITE / Turn task-branch changes into a reviewable proposal.')
    step(1, 'Click <b>Compare &amp; pull request</b> after committing, or use <b>Pull requests &gt; New pull request</b>. Select <b>base: main</b> and <b>compare: docs/experiment-guide</b>.')
    step(2, 'Read the proposed diff. Add a clear title and description, then click <b>Create pull request</b>. Use a draft if available for unfinished work; otherwise label it clearly as not ready to merge.')
    step(3, 'Request a teammate in <b>Reviewers</b> if available, or ask in the PR discussion. The reviewer opens <b>Files changed</b>, reads the diff and uses <b>Review changes</b> to comment, approve or request changes.')
    step(4, 'Address comments by editing the <b>same source branch</b> and committing again. The existing PR updates. Recheck the branch selector when navigating from a PR back to the file editor.')
    h('Copyable description')
    code('Purpose\nExplain the experiment setup for new teammates.\n\nChanges\n- Describe what changed and why.\n\nValidation\n- State exactly what you checked.\n- State what you did not run and why.\n\nKnown limitations\n- List remaining work or missing information.')
    note('Review is more than clicking Approve', 'Check correctness, scope and validation evidence. A browser preview does not execute code. If important changes follow approval, ask for another review. The team recommends one other reviewer; enforcement depends on settings.')

    page('Merge and finish', 'WEBSITE / The maintainer coordinates initial team merges.')
    step(1, 'The maintainer verifies the PR targets main, review comments are addressed, and required checks have passed. For a small task, the team can standardize on <b>Squash and merge</b> if enabled.')
    step(2, 'After confirmation, check that the PR says <b>Merged</b>. Return to <b>Code</b>, select <b>main</b>, and verify the resulting file. Refresh the page if necessary.')
    step(3, 'Use <b>Delete branch</b> on the merged PR only when nobody needs that branch for more work. Start the next task on a fresh branch from main.')
    diagram(['PR merged', 'View main', 'New branch'], ['On GitHub', 'Check final result', 'For the next task'])
    note('Website-only work has no local pull step', 'Once the PR is merged, selecting main on GitHub shows the server-side result. A copy of the project already downloaded to your Mac is separate and must be updated.')
    h('Optional: update an existing local clone')
    p('From the local repository, first preserve any unfinished edits and ensure the working tree is clean. Then:')
    code('git switch main\ngit pull --ff-only origin main')
    p('Do not reuse a merged task branch for another task. After squash merging, local git branch -d may refuse deletion because the original commits are not ancestors of main. Leave the branch until its contents are verified; do not force-delete just to remove the warning.', 'small')
    h('If the PR is closed but not merged')
    p('Closed and Merged are different states. A closed, unmerged PR did not add its changes to main. Read the discussion before restarting or reopening the work.')

    page('Update a PR with main', 'WEBSITE / Refreshing a page does not merge main into your task branch.')
    p('A teammate merged changes while your PR was open. Your task branch may now be behind main. Integrating those changes helps you check compatibility before your own PR is merged.')
    step(1, 'Open your PR and confirm <b>base: main</b> and the correct source branch. Scroll to the merge section near the bottom.')
    step(2, 'If GitHub offers <b>Update branch</b>, use its merge option. This brings main into your PR branch. Avoid <b>Update with rebase</b> in this beginner workflow unless the maintainer coordinates it.')
    step(3, 'Read the updated diff and rerun affected checks in the appropriate execution environment. The update is already on GitHub.')
    step(4, 'If conflicts are reported, follow page 15. If no Update branch control appears, it may not be needed or available under current settings. Ask the maintainer; do not create a reverse PR into main as a workaround.')
    table(['Action', 'What it does'], [('Select main in the website', 'Displays main; does not change the task branch.'), ('Update branch on the task PR', 'Integrates base main into the source branch when available.'), ('Merge the reviewed PR', 'Integrates the task result into main.'), ('Pull in a local clone', 'Updates that local branch from its configured remote counterpart.')])
    h('Optional local fallback - start with a clean working tree')
    code('git fetch origin\ngit switch docs/experiment-guide\ngit pull --ff-only origin docs/experiment-guide\ngit merge origin/main\n# Resolve conflicts and validate, then:\ngit push')
    p('If the fast-forward-only pull fails, preserve the work and inspect the divergence with the maintainer. Do not force-push.', 'small')

    page('Website and local Mac work', 'Use one editing location at a time for the same branch.')
    h('If you work only on the website')
    p('Finish a small commit before switching browser tabs, branches or computers. Uncommitted editor text is not a shared Git commit; copy valuable drafts elsewhere before navigating away. Reopen the branch on another device to see committed work.')
    h('Continue a website branch on your Mac')
    p('Set up a clone using page 23. From a clean working tree, fetch and create a local tracking branch for the published branch. The tracking command below is for a branch that does not already exist locally.')
    code('cd ~/Documents/STS2026\ngit fetch origin\ngit switch --track origin/docs/experiment-guide')
    p('If it already exists locally, use git switch docs/experiment-guide and git pull --ff-only instead. Run the documented project checks, then commit and push your intended changes.')
    h('Return to editing on the website')
    p('Push local commits first and verify them on GitHub. After later website commits, pull that same branch before resuming local edits. Browser commits and local commits can diverge if both locations change independently.')
    code('git status\ngit add -- docs/experiment-guide.md\ngit diff --cached\ngit commit -m "docs: clarify experiment setup"\ngit push')
    note('Pause unfinished local work', 'Either make a clearly labelled checkpoint commit on your task branch, or use a local stash with help. git stash push -u includes untracked files but not ignored files. A stash is local and is not a GitHub backup. Preserve it until restored changes are checked.')
    p('In macOS Terminal, ~ means your home directory. Teammates may choose another local folder; never use the maintainer\'s Windows drive path on your Mac.', 'small')

    page('Notebooks and experiments', 'WEBSITE FOR REVIEW / A real execution environment for running code.')
    p('The project includes vits-light_v1.ipynb. GitHub can display notebook content, but its repository page does not run notebook cells. Editing notebook JSON in a text editor is fragile; use a notebook-aware local or team environment.')
    step(1, 'Agree a primary editor for each notebook task. Coordinate before anyone else modifies the same notebook; whole-file uploads can overwrite another person\'s changes.')
    step(2, 'Start from the latest intended branch. Run a small documented check before changing code. For substantive notebook work, prefer a local clone so you can inspect and commit the actual diff.')
    step(3, 'Before committing, inspect cell outputs, metadata and paths. Remove disposable bulk output and secrets while preserving the evidence needed for the report.')
    step(4, 'Restart the kernel and execute the relevant cells in order where practical. In the PR, distinguish a small sample, import check and full experiment. Do not mark an unrun change as tested.')
    table(['Record', 'Purpose'], [('Code commit + local edits, if any', 'Identify the code that actually ran.'), ('Configuration and random seed', 'Make comparisons repeatable.'), ('Data version and split', 'Keep evaluation comparable.'), ('Environment and compute', 'Explain dependency and runtime differences.'), ('Metrics and artifact location', 'Connect reported results to evidence.')])
    note('Mac hardware is not automatically the training environment', 'A project that needs NVIDIA CUDA cannot use a Mac\'s Apple GPU as a drop-in replacement. Follow the project\'s documented CPU, supported Mac or remote GPU setup. Do not promise training support until it is verified.')
    p('For notebook conflicts, preserve both versions, agree a base and reapply intended cell changes with the other editor. Verify the resolved notebook opens and runs. Do not blindly accept an entire side.', 'small')

    page('What belongs in Git?', 'Review uploaded files as carefully as locally committed files.')
    table(['Track and review', 'Keep outside normal Git history'], [('Source, README, dependency definitions', 'Virtual environments, caches and scratch files'), ('Placeholder-only configuration examples', 'Real .env files, passwords and tokens'), ('Small curated metrics and report figures', 'Datasets, checkpoints and bulk training outputs'), ('Data access and version instructions', 'Private material and machine-specific paths')])
    h('Suggested .gitignore starting point')
    code('.DS_Store\n.env\n.venv/\nvenv/\n__pycache__/\n.ipynb_checkpoints/\ntmp/\ndata/\ndatasets/\ncheckpoints/\n*.pt\n*.pth\n*.ckpt')
    note('Web uploads still need manual review', '.gitignore guides local Git file selection; do not rely on it to prevent manually selected website uploads. It also does not remove already tracked files or erase previous commits.')
    p('Do not ignore all of output/ automatically. A selected report figure or this PDF may belong in Git; large temporary artifacts should have a separate location. On macOS, .DS_Store stores Finder metadata and should normally be excluded.')
    p('Share datasets and checkpoints through team-agreed storage with access instructions and versions. Git LFS requires deliberate configuration and storage planning; adding it later does not automatically clean old Git history.', 'small')

    page('Access and upload problems', 'WEBSITE FIRST / Identify the account, repository and branch before retrying.')
    table(['Symptom', 'What to do'], [('Repository not found / 404', 'Check the exact URL, active account and accepted invitation. A private repo can look missing if you lack access.'), ('Cannot create a branch or edit', 'Have the maintainer confirm collaborator access and branch rules. Watching or starring the repo does not grant write access.'), ('GitHub proposes a fork', 'For this shared-repo workflow, check that you are signed in to the invited account with write access.'), ('Upload fails or file is too large', 'Browser uploads allow 25 MiB per file and 100 files per batch. Do not split a model into chunks to bypass the project\'s storage rules.'), ('Cannot commit a stale web edit', 'Copy your draft first. Reload the current branch version and reconcile changes; do not overwrite another person\'s newer work.'), ('PR cannot merge', 'Read whether it needs approval, checks, branch updates or conflict resolution. Page 15 covers conflicts.'), ('Empty repo / no main branch', 'The maintainer must publish the baseline; see page 4.')], [150, 353])
    h('Optional local Git errors')
    p('For authentication failure, reauthenticate the affected client; a browser login does not automatically configure Terminal. For non-fast-forward / fetch first, preserve work, fetch and inspect divergence. Do not force-push.')
    note('Ask for useful help', 'Share the URL, branch, action and exact error, with secrets removed. Say whether you edited in the browser or locally. For local issues include git status. This tells the maintainer which copy needs attention.')

    page('Resolve a web conflict', 'WEBSITE / Use this for simple text conflicts, not notebook JSON or binary files.')
    step(1, 'Open the task PR. Read the conflict warning and click <b>Resolve conflicts</b> if enabled. Confirm the base is main and the head is your task branch.')
    step(2, 'Read both versions and agree the intended result with the other author. Use the branch labels beside each conflict to identify the sides; do not infer ownership from top/bottom position alone.')
    code('<<<<<<< docs/experiment-guide\nRecord the sample list.\n=======\nRecord the random seed.\n>>>>>>> main')
    step(3, 'Edit one correct combined result and remove the conflict markers. Check all conflicts in every listed file. Click <b>Mark as resolved</b> for each completed file.')
    step(4, 'Click <b>Commit merge</b> and confirm the intended head branch. This merges the entire base branch into the task branch; it does not finish merging the task PR into main.')
    step(5, 'Inspect the updated diff and run the relevant checks in the execution environment. Ask for review again if needed; the maintainer can then merge the PR.')
    note('Resolve conflicts is disabled?', 'Some conflicts require a local Git client. Preserve your work and ask the maintainer to help. For notebooks, preserve both versions and resolve cell changes in a notebook-aware environment. For PDFs/images, agree a version or regenerate from source.')
    h('If a local merge must be abandoned')
    p('git merge --abort applies to an in-progress local merge, not a website commit that already exists. It can discard resolution edits; use it only with a understood clean pre-merge state and preserve anything valuable first.')

    page('I used the wrong branch', 'A completed website commit is already shared on GitHub.')
    table(['State', 'Recovery'], [('Still editing in the browser; not committed', 'Copy your draft somewhere safe. Cancel/navigate back, select or create the correct task branch, reopen the current file and reapply the intended edits.'), ('Website commit landed on main', 'Tell the maintainer immediately. Decide whether it can stay or needs a corrective/revert PR. There is no unpublished local commit to undo.'), ('Website commit landed on another task branch', 'Record the commit link and notify that branch owner. Coordinate moving the intended change and correcting the original branch without losing their work.'), ('Local uncommitted edits on main', 'Create a rescue/task branch before committing. Review which edits came with you.'), ('Local main commits not pushed', 'Create a rescue branch to preserve the commits. Have the maintainer inspect history before restoring main.')], [150, 353])
    h('Local-only first step, when preserving a local commit or edit')
    code('git status\ngit switch -c fix/rescue-work')
    p('This creates a branch name for your current state; it does not reset main. Review and commit uncommitted edits on the rescue branch if needed. Do not use this as a remedy for an already published web commit.')
    note('Prevent the next mistake', 'Check the branch selector before opening the editor and check the commit destination again in the Commit changes dialog. Multiple browser tabs may point to different branches.')

    page('Undo or recover a file', 'WEBSITE / Prefer an explicit corrective change that preserves shared history.')
    h('A small incorrect website edit')
    step(1, 'Record the bad commit or PR link. Create a fresh correction branch from main if the change is already merged, or correct the existing task branch if its PR is still open.')
    step(2, 'Open the file history to inspect the last correct version. Reapply only the needed correction to the current file; replacing it wholesale may erase later valid changes.')
    step(3, 'Commit the correction, explain it in the PR and request review. The earlier bad commit remains in history; the correction records what changed.')
    h('Revert a merged PR')
    p('For a merged PR, the maintainer can use <b>Revert</b> if GitHub offers it. GitHub prepares a new revert PR; inspect, validate and review it before merging. If the button is unavailable or conflicts occur, coordinate a local revert. Do not assume every merge type has the same recovery steps.')
    h('Restore a deleted file')
    p('Find the last intended version through file/commit history, then recreate it at the correct path on a recovery branch and open a PR. Review whether other code depends on its deletion. A never-committed local file cannot be recovered from Git history.')
    note('Local-only undo is a different operation', 'GitHub Desktop can undo a latest unpushed local commit. That does not apply to github.com commits, which are already remote. Discarding uncommitted edits can lose work that Git never recorded.', True)
    h('Before copying old content')
    p('Keep the current version, identify the exact change you want to reverse, and inspect later edits. Never use force push or hard reset as a routine way to undo a shared mistake.')

    page('Wrong files or secrets', 'Deleting a file in a later commit does not erase its earlier contents.')
    h('An unintended upload')
    step(1, '<b>Before committing:</b> remove the file from the upload selection or cancel and restart with the correct selection. Check every filename and destination.')
    step(2, '<b>After a website commit:</b> it is already in remote history. Tell the maintainer. A deletion PR stops the file appearing in the latest snapshot, but not in previous commits.')
    step(3, '<b>Large files or older local commits:</b> ask the maintainer to inspect the history before retrying. A fresh deletion does not fix a large blob already inside an earlier commit.')
    h('If a credential was uploaded')
    step(4, '<b>Revoke or rotate the credential with its provider first.</b> Treat it as exposed even in a private repository. Deleting the file alone cannot invalidate the credential.')
    step(5, 'Tell the maintainer the credential type and location without reposting its value. Check notebook outputs, logs and PR diffs too. Coordinate history cleanup and updates to existing clones if required.')
    note('Repository visibility is not cleanup', 'Making a repository private or deleting a branch does not revoke a credential or guarantee all copies are removed. Follow the official sensitive-data guidance with the maintainer.', True)
    h('Prevent repeats')
    p('Use placeholder-only examples, keep real configuration outside Git and inspect selected upload files. Avoid uploading whole Finder folders without review. For local-only tracked files, stopping tracking changes future snapshots; it still does not purge history.')

    page('Your first browser-only PR', '15-20 MINUTES / No Terminal or GitHub Desktop required.')
    step(1, 'Accept the invite using page 5. Open STS2026 in Safari or Chrome and select main after the maintainer publishes the baseline.')
    step(2, 'Create <b>docs/onboarding-YOUR_USERNAME</b> from main. Replace YOUR_USERNAME with your GitHub username.')
    step(3, 'On that branch choose <b>Add file &gt; Create new file</b>. Name it <b>docs/team/YOUR_USERNAME.md</b>. Add your preferred name, intended project role and one setup question. Avoid private contact details.')
    step(4, 'Preview the text and click <b>Commit changes</b>. Confirm the commit goes to your task branch. Then open a PR with <b>base: main</b>.')
    step(5, 'Have a teammate request one useful clarification in review. Edit the same file on the same task branch and commit again. Confirm the existing PR updates.')
    step(6, 'After approval, the maintainer merges. Select main on the repository page and confirm your note appears. Create a new branch for your first real task.')
    diagram(['Author', 'Reviewer', 'Maintainer'], ['Creates + revises', 'Checks + comments', 'Merges'])
    note('Finished means', 'The PR says Merged, main contains your note, and you can explain why a website commit needs a PR but no separate push. Report any unclear step so the handbook can improve.')
    p('The exercise verifies your collaboration workflow, not the model or experiment pipeline. Actual execution checks belong in the project setup process.', 'small')

    page('Maintainer playbook', 'Recommended team policy; repository settings must be configured separately.')
    h('Invite each teammate in person')
    p('Collect the GitHub username/profile link. Open <b>Settings &gt; Collaborators &gt; Add people</b>, check the matching account, and send the invitation. Have the person complete page 5. Confirm each invite changes from pending to accepted.')
    h('Use a small, clear workflow')
    p('Keep main plus short-lived task branches. Require a PR and one other reviewer by team agreement. Assign one owner per task and one primary editor per notebook. Initially let the maintainer coordinate merges.')
    h('Enforce rules where supported')
    p('Confirm main is the default branch. Where the plan supports it, require PRs, approval and resolved conversations; disallow force pushes and deletion of main. Check bypass permissions. Private-repository branch protection depends on the account/organization plan.')
    note('Conventions and enforcement differ', 'If the current plan cannot enforce a rule, keep it explicit and coordinate merges manually. This handbook does not certify which protections are enabled.')
    h('Document and test real requirements')
    table(['Item', 'Minimum useful content'], [('README.md', 'Project goal, verified macOS/remote setup, dependencies and data access.'), ('CONTRIBUTING.md', 'Branch, PR and review rules; links to this handbook.'), ('Configuration example', 'Documented placeholder values only.'), ('Validation', 'Small meaningful checks; actual experiment evidence and limitations.')])
    p('Do not require a nonexistent automated check or claim a smoke test validates full training. At a report milestone, record the reviewed code commit and corresponding experiment artifacts.', 'small')

    page('References and glossary', 'Official GitHub documentation / checked 17 September 2026.')
    refs = [
        ('Accepting a collaborator invitation', 'repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository'),
        ('Creating branches on GitHub', 'pull-requests/how-tos/commit-changes/managing-branches-within-your-repository'),
        ('Editing files on the website', 'repositories/working-with-files/managing-files/editing-files'),
        ('Uploading files and size limits', 'repositories/working-with-files/managing-files/adding-a-file-to-a-repository'),
        ('Updating a PR with its base branch', 'pull-requests/how-tos/create-pull-requests/keeping-your-pull-request-in-sync-with-the-base-branch'),
        ('Resolving conflicts on GitHub', 'pull-requests/how-tos/merge-and-close-pull-requests/resolving-a-merge-conflict-on-github'),
        ('Reverting a pull request', 'pull-requests/how-tos/merge-and-close-pull-requests/reverting-a-pull-request'),
        ('Removing sensitive data', 'authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository'),
        ('The github.dev editor and its limitations', 'codespaces/the-githubdev-web-based-editor'),
        ('Branch protection and plan availability', 'repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches'),
    ]
    for i, (label, path) in enumerate(refs, 1):
        p(f'<b>{i}.</b> <a href="https://docs.github.com/en/{path}" color="#007F82">{label}</a>')
    h('Small glossary')
    table(['Term', 'Meaning'], [('Base / head', 'PR destination / source branch, usually main / your task branch.'), ('Working tree / staging area', 'Local checked-out files / changes selected for the next local commit.'), ('HEAD / origin/main', 'Current checked-out commit / local record of remote main as of the last fetch.'), ('Push / pull', 'Upload local commits / integrate remote changes into a local branch.'), ('github.dev / Codespaces', 'Browser editor / separate cloud execution environment. They are not interchangeable.')], [125, 378])
    p('Project: <a href="https://github.com/liquidtoy001/STS2026" color="#007F82">github.com/liquidtoy001/STS2026</a>. Site labels may change. Example paths and validation steps must be adapted to the actual project.', 'small')

    page('Website desk reference', 'OPEN / BRANCH / EDIT / COMMIT / PR / REVIEW / MERGE')
    table(['Moment', 'Check'], [('Join', 'Correct GitHub account? Invitation accepted? Initial main published?'), ('Start', 'Select main. Create a fresh task branch. Confirm the branch name.'), ('Edit', 'Use the pencil or Add file. Check the path and preview the change.'), ('Commit', 'Specific message? Correct task branch? No secrets or bulk uploads?'), ('PR', 'Base main, correct source branch, clear purpose, honest validation, reviewer.'), ('Finish', 'PR says Merged. Select main and verify. Start a fresh branch next time.')])
    diagram(['Task branch', 'Web commit', 'PR into main'], ['Check selector', 'Already remote', 'Needs review'])
    h('Three common misunderstandings')
    p('<b>No push after a normal website commit.</b> It is already on GitHub.<br/><b>Refreshing is not branch integration.</b> Use the PR update workflow on page 10.<br/><b>Viewing is not running.</b> Execute code in the documented environment, then report what actually ran.')
    h('Where to look when stuck')
    p('Invitation: page 5. Missing edit/upload access: page 14. Conflict: page 15. Wrong branch: page 16. Undo/recovery: page 17. Secrets/large files: page 18. Local Mac setup: page 23.')
    note('When asking for help', 'Send the page/PR link, branch name, intended action and exact error. Say whether you were using github.com, github.dev or a local clone. Remove passwords and tokens from screenshots.')

    page('Optional: local Mac setup', 'Only needed when your task requires local editing, execution or Git recovery.')
    h('Option A: GitHub Desktop for macOS')
    step(1, 'Install GitHub Desktop for your Mac and sign in to the invited account. Configure your author name and GitHub-associated email (or GitHub private commit email) in its Git settings.')
    step(2, 'Choose <b>File &gt; Clone Repository &gt; URL</b> and paste the STS2026 URL below. Select a new/empty destination such as <b>~/Documents/STS2026</b>. Open it in your editor.')
    code('https://github.com/liquidtoy001/STS2026.git')
    h('Option B: Terminal, if Git is installed')
    p('Open Terminal with Spotlight (Command-Space, then type Terminal). Check git --version; if macOS prompts for Command Line Tools, complete that installation first. Then, from a parent folder:')
    code('cd ~/Documents\ngit clone https://github.com/liquidtoy001/STS2026.git\ncd STS2026\ngit remote -v\ngit status')
    p('For a private repository, HTTPS cloning needs working Git authentication. Your GitHub account password is not accepted as a Git HTTPS password. Use Desktop\'s sign-in flow, or a documented credential-manager/PAT or SSH setup. Never paste a token into a shared command or repository URL.')
    note('If ~/Documents/STS2026 already contains work', 'Do not clone over it. Choose another empty destination and preserve any existing edits. Download ZIP is fine for reading a snapshot but does not create a Git-connected working copy.')
    h('Before running the project')
    p('Follow the actual README for Python, dependencies, data and compute. Cloning does not install them. Apple Silicon, Intel Macs and GPU servers may need different environments. See page 12 for notebook validation, and page 11 for switching between local and web editing.')
