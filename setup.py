from setuptools import setup , find_packages

def get_requirements(file_path:str)->list[str]:
    with open(file_path) as f:
        reqs=f.readlines()
    reqs=[r.strip() for r in reqs]
    reqs=[r for r in reqs if r and not r.startswith("#")]
    if "-e ." in reqs:
        reqs.remove("-e .")
    return reqs

setup(
    name="debateedge",
    version="0.1.0",
    author="Kafil Aslam",
    author_email="aslamkafil13@gmail.com",
    description="AI Debate and Argument Coach - Production Grade Agent System",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.12"
)