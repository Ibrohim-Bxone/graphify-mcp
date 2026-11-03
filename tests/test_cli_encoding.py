"""Windows' legacy code page must not break the CLI's UTF-8 protocol."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


CLI=next((parent/'graphify-cli.py' for parent in Path(__file__).resolve().parents
          if (parent/'graphify-cli.py').is_file()),None)
pytestmark=pytest.mark.skipif(CLI is None,reason='Workspace CLI is not part of a standalone Graphify checkout')


def invoke(arguments,source,input=None):
    stub='''import runpy,sys,types
sys.modules['server']=types.SimpleNamespace(list_memories=lambda limit:'🔥 O‘zbekcha')
sys.modules['keyword_index']=types.SimpleNamespace(get_index=lambda:None)
sys.modules['db']=types.SimpleNamespace(get_collection=lambda:None)
sys.modules['search_service']=types.SimpleNamespace(factory_context=lambda query,project,**kw:{'query':query,'answer':'🔥 O‘zbekcha'})
'''
    script=stub+source
    env=dict(os.environ,PYTHONIOENCODING='cp1251')
    result=subprocess.run([sys.executable,'-c',script,str(CLI),*arguments],input=input,capture_output=True,env=env,timeout=15)
    assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
    return result.stdout.decode('utf-8')


def test_cli_unicode_output_ignores_windows_legacy_code_page():
    output=invoke(['list'],"sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
    assert output.strip()=='🔥 O‘zbekcha'


def test_cli_stdin_and_json_stdout_share_utf8_contract():
    query='O‘zbekcha bilimlar 🔎'
    output=invoke(['context','--stdin','--project','test'],
                  "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')",query.encode('utf-8'))
    assert json.loads(output)=={'query':query,'answer':'🔥 O‘zbekcha'}
