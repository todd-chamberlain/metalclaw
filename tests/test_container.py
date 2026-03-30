"""Tests for container build argument and runtime validation."""

import pytest

from metalclaw.container import (
    _validate_agent_command,
    _validate_base_image,
    _validate_env_var,
    _validate_pip_requirement,
    _validate_system_packages,
    _validate_url_scheme,
)


# ---------------------------------------------------------------------------
# base_image validation
# ---------------------------------------------------------------------------

class TestValidateBaseImage:
    def test_fedora_default(self):
        assert _validate_base_image("registry.fedoraproject.org/fedora:42") is None

    def test_ubuntu(self):
        assert _validate_base_image("ubuntu:24.04") is None

    def test_rhel_ubi(self):
        assert _validate_base_image("registry.access.redhat.com/ubi9/ubi") is None

    def test_alpine(self):
        assert _validate_base_image("alpine:3.20") is None

    def test_opensuse(self):
        assert _validate_base_image("opensuse/tumbleweed") is None

    def test_azure_linux(self):
        assert _validate_base_image("mcr.microsoft.com/azurelinux/base/core:3.0") is None

    def test_digest_reference(self):
        assert _validate_base_image("ubuntu@sha256:abcdef1234567890") is None

    def test_empty_rejected(self):
        assert _validate_base_image("") is not None

    def test_shell_injection_rejected(self):
        assert _validate_base_image("$(whoami)@evil.com/image") is not None

    def test_semicolon_rejected(self):
        assert _validate_base_image("ubuntu:24.04; rm -rf /") is not None

    def test_backtick_rejected(self):
        assert _validate_base_image("`whoami`") is not None

    def test_pipe_rejected(self):
        assert _validate_base_image("ubuntu | cat /etc/passwd") is not None

    def test_starts_with_dash_rejected(self):
        assert _validate_base_image("-evil/image:latest") is not None


# ---------------------------------------------------------------------------
# extra_system_packages validation
# ---------------------------------------------------------------------------

class TestValidateSystemPackages:
    def test_empty_is_valid(self):
        assert _validate_system_packages("") is None

    def test_single_package(self):
        assert _validate_system_packages("vim") is None

    def test_multiple_packages(self):
        assert _validate_system_packages("vim htop strace") is None

    def test_package_with_version_chars(self):
        assert _validate_system_packages("g++ libfoo-dev") is None

    def test_semicolon_rejected(self):
        assert _validate_system_packages("vim; rm -rf /") is not None

    def test_dollar_rejected(self):
        assert _validate_system_packages("vim $(whoami)") is not None

    def test_backtick_rejected(self):
        assert _validate_system_packages("vim `id`") is not None

    def test_pipe_rejected(self):
        assert _validate_system_packages("vim | cat") is not None

    def test_ampersand_rejected(self):
        assert _validate_system_packages("vim && echo pwned") is not None

    def test_redirect_rejected(self):
        assert _validate_system_packages("vim > /tmp/x") is not None


# ---------------------------------------------------------------------------
# ca_cert_url validation (HTTPS-only)
# ---------------------------------------------------------------------------

class TestValidateUrlScheme:
    def test_empty_is_valid(self):
        assert _validate_url_scheme("") is None

    def test_https_valid(self):
        assert _validate_url_scheme("https://pki.corp.com/ca.pem") is None

    def test_http_rejected(self):
        """CA certs over HTTP are vulnerable to MITM — must use HTTPS."""
        assert _validate_url_scheme("http://internal.corp/ca.pem") is not None

    def test_file_protocol_rejected(self):
        assert _validate_url_scheme("file:///etc/passwd") is not None

    def test_ftp_rejected(self):
        assert _validate_url_scheme("ftp://evil.com/cert.pem") is not None

    def test_gopher_rejected(self):
        assert _validate_url_scheme("gopher://evil.com") is not None

    def test_dict_rejected(self):
        assert _validate_url_scheme("dict://evil.com") is not None

    def test_no_scheme_rejected(self):
        assert _validate_url_scheme("pki.corp.com/ca.pem") is not None


# ---------------------------------------------------------------------------
# deploy.extra_env validation
# ---------------------------------------------------------------------------

class TestValidateEnvVar:
    def test_normal_env_var(self):
        assert _validate_env_var("MY_SETTING", "value") is None

    def test_underscore_prefix(self):
        assert _validate_env_var("_INTERNAL", "x") is None

    def test_ld_preload_blocked(self):
        assert _validate_env_var("LD_PRELOAD", "/evil.so") is not None

    def test_ld_library_path_blocked(self):
        assert _validate_env_var("LD_LIBRARY_PATH", "/tmp") is not None

    def test_path_blocked(self):
        assert _validate_env_var("PATH", "/evil:/usr/bin") is not None

    def test_pythonpath_blocked(self):
        assert _validate_env_var("PYTHONPATH", "/tmp") is not None

    def test_pythonstartup_blocked(self):
        assert _validate_env_var("PYTHONSTARTUP", "/evil.py") is not None

    def test_node_options_blocked(self):
        assert _validate_env_var("NODE_OPTIONS", "--require /evil.js") is not None

    def test_bash_env_blocked(self):
        assert _validate_env_var("BASH_ENV", "/evil.sh") is not None

    def test_http_proxy_blocked(self):
        assert _validate_env_var("http_proxy", "http://evil.com") is not None

    def test_https_proxy_blocked(self):
        assert _validate_env_var("HTTPS_PROXY", "http://evil.com") is not None

    def test_dyld_insert_blocked(self):
        assert _validate_env_var("DYLD_INSERT_LIBRARIES", "/evil.dylib") is not None

    def test_invalid_key_format(self):
        assert _validate_env_var("123BAD", "val") is not None

    def test_key_with_special_chars(self):
        assert _validate_env_var("MY-VAR", "val") is not None

    def test_key_with_spaces(self):
        assert _validate_env_var("MY VAR", "val") is not None


# ---------------------------------------------------------------------------
# pip requirement validation
# ---------------------------------------------------------------------------

class TestValidatePipRequirement:
    def test_simple_package(self):
        assert _validate_pip_requirement("requests") is None

    def test_pinned_version(self):
        assert _validate_pip_requirement("websockets<14") is None

    def test_version_range(self):
        assert _validate_pip_requirement("flask>=2.0,<3.0") is None

    def test_file_url_rejected(self):
        assert _validate_pip_requirement("file:///tmp/evil.tar.gz") is not None

    def test_git_url_rejected(self):
        assert _validate_pip_requirement("git+https://evil.com/repo.git") is not None

    def test_hg_url_rejected(self):
        assert _validate_pip_requirement("hg+https://evil.com/repo") is not None

    def test_svn_url_rejected(self):
        assert _validate_pip_requirement("svn+https://evil.com/repo") is not None

    def test_index_url_rejected(self):
        assert _validate_pip_requirement("pkg --index-url http://evil.com") is not None

    def test_extra_index_url_rejected(self):
        assert _validate_pip_requirement("pkg --extra-index-url http://evil.com") is not None

    def test_find_links_rejected(self):
        assert _validate_pip_requirement("pkg --find-links /tmp") is not None

    def test_trusted_host_rejected(self):
        assert _validate_pip_requirement("pkg --trusted-host evil.com") is not None


# ---------------------------------------------------------------------------
# agent command validation
# ---------------------------------------------------------------------------

class TestValidateAgentCommand:
    def test_simple_command(self):
        _validate_agent_command("/usr/bin/python3 script.py")  # should not raise

    def test_command_with_args(self):
        _validate_agent_command("node /opt/agent/index.js --port 8080")

    def test_dollar_paren_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("$(whoami)")

    def test_backtick_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("`id`")

    def test_double_ampersand_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd1 && cmd2")

    def test_double_pipe_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd1 || cmd2")

    def test_semicolon_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd1; cmd2")

    def test_pipe_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd | cat")

    def test_redirect_out_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd > /tmp/x")

    def test_redirect_in_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd < /etc/passwd")

    def test_newline_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("cmd\n/bin/evil")

    def test_glob_star_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("/usr/bin/python3 *")

    def test_glob_question_rejected(self):
        with pytest.raises(ValueError):
            _validate_agent_command("/usr/bin/python3 ?")
