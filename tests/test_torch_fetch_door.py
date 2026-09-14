"""The torch half of the offline door must be shut, and something must watch it.

`conftest.py` has forced `HF_HUB_OFFLINE` and friends for a while, and that closed the
HuggingFace door. Torch has its own, and `TORCH_HOME` only RELOCATES torchvision's cache
rather than closing it — so `download.pytorch.org` stayed open, `test_model_instantiates`
downloaded silently on a networked runner, and it passed.

That is how `deeplab.py` shipped calling `deeplabv3_resnet50(pretrained=False, ...)` while
its `weights_backbone` default fetched a 97.8 MB ResNet-50 checkpoint on every construction
(#289). Nothing was watching the door, so nothing reported it opening.

This file watches it, for the reason `test_dump_fetch_guard.py` gives about its own subject:
a guard whose branch nobody asserts is a guard that can be deleted, or quietly stop working,
without a single test turning red.
"""

import pytest

# Same reason conftest guards its patch: the sklearn and survival matrices have no torch,
# and a module-level import here took their whole collection down. `importorskip` skips
# only where torch is genuinely absent, so in every matrix that HAS torch these still run
# and still fail if the door is open.
#
# `torch` and nothing narrower (@saadqbal ask 2). This file used to
# `importorskip("torch.utils.model_zoo")` at module level, which handed the DEPRECATED
# alias shim a veto over all three watchers: the day torch drops it, every test here
# SKIPS — green suite, open door — which is the exact failure mode a watchdog must not
# have. The alias now gets skipped in its own test and nowhere else.
pytest.importorskip("torch")

import torch.hub  # noqa: E402


def test_load_state_dict_from_url_is_refused():
    """A template reaching torch's URL loader must fail, not download."""
    with pytest.raises(AssertionError) as excinfo:
        torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/x.pth")
    assert "fetched pretrained weights at build time" in str(excinfo.value)


def test_model_zoo_load_url_alias_is_refused():
    """`torch.utils.model_zoo.load_url` IS the same function; the alias must be shut too.

    Skipped HERE and only here if torch ever drops the deprecated shim, so its absence
    costs this one assertion instead of silencing the whole file.
    """
    torch_model_zoo = pytest.importorskip("torch.utils.model_zoo")
    with pytest.raises(AssertionError):
        torch_model_zoo.load_url("https://download.pytorch.org/models/x.pth")


def test_the_refusal_names_the_actual_fix():
    """The message has to say `weights_backbone=None`, not `pretrained=False`.

    `pretrained=False` is the spelling that looked like it settled this and did not: the
    legacy shim maps it to `weights=None` and never touches `weights_backbone`.

    ONE token, not two prose substrings (@LukasWodka): `weights_backbone=None` is the
    load-bearing thing a reader must find. Pinning the surrounding sentence as well made
    this a copy of the message, red on a reword that changed nothing about the door.
    """
    with pytest.raises(AssertionError, match=r"weights_backbone=None"):
        torch.hub.load_state_dict_from_url("https://example.invalid/x.pth")


def test_the_real_builder_is_refused_not_just_the_patched_attribute():
    """Build the template's own call and watch the door stop it.

    @saadqbal ask 1. The three watchers above call the REBOUND attribute, so what they
    assert is that the patch happened — not that the fetch path is shut. Those are the
    same statement only while nothing has already captured the original function, and
    torchvision captures it by VALUE at import:

        from torch.hub import load_state_dict_from_url

    so a module imported before `conftest` patched would hold the real downloader and
    every assertion above would still pass. Today the ordering is in our favour (one
    conftest, no pytest plugin config, so the patch lands first), and that is a property
    of the current layout rather than of the door.

    This test does not depend on it. It constructs `deeplabv3_resnet50(weights=None)` —
    #289's exact shipped call, whose `weights_backbone` still defaults to
    `ResNet50_Weights.IMAGENET1K_V1` — and asserts the construction is REFUSED. If the
    binding is ever captured early the download resumes and this test is what turns red.
    """
    models = pytest.importorskip("torchvision.models")

    with pytest.raises(AssertionError) as excinfo:
        models.segmentation.deeplabv3_resnet50(weights=None)
    assert "fetched pretrained weights at build time" in str(excinfo.value)


def test_the_fixed_call_constructs_offline():
    """The pattern the refusal recommends has to actually work.

    The other half of the test above: a message telling the next person to pass
    `weights=None` AND `weights_backbone=None` is only useful if that really does build
    without touching the network. This is `deeplab.py`'s call after #289.
    """
    models = pytest.importorskip("torchvision.models")

    model = models.segmentation.deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=3
    )
    assert model is not None


def test_the_socket_is_shut_too():
    """The layer that does not depend on import order (@LukasWodka ask 3).

    `prep_offline_weights._block_network()` refuses at the socket, which is what covers
    `torch.hub.download_url_to_file` (timm's fetch path), `urllib` and `requests` --
    none of which a name-rebind of `load_state_dict_from_url` can see, and none of which
    care who imported torchvision first. Removing the `_block_network()` call from
    conftest reddens this and nothing else, which is how it is meant to fail.
    """
    import socket

    with pytest.raises(RuntimeError, match=r"network access blocked"):
        socket.getaddrinfo("download.pytorch.org", 443)
