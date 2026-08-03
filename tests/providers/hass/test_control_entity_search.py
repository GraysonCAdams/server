"""Tests for the Home Assistant player control entity search."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import Scope
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import CONF_LOG_LEVEL
from music_assistant.helpers.api import APICommandHandler
from music_assistant.providers.hass import (
    CONF_AUTH_TOKEN,
    CONF_URL,
    CONF_VERIFY_SSL,
    SEARCH_CONTROL_ENTITIES_COMMAND,
    HomeAssistantProvider,
    setup,
)
from music_assistant.providers.hass.constants import (
    CONF_MUTE_CONTROLS,
    CONF_POWER_CONTROLS,
    CONF_VOLUME_CONTROLS,
    MediaPlayerEntityFeature,
)
from music_assistant.providers.hass.control_entities import (
    HassControlEntityGroup,
    HassControlEntitySearchResult,
)

REGISTRY_LIST_COMMAND = "config/entity_registry/list_for_display"

FULL_MEDIA_PLAYER_FEATURES = int(
    MediaPlayerEntityFeature.TURN_ON
    | MediaPlayerEntityFeature.TURN_OFF
    | MediaPlayerEntityFeature.VOLUME_SET
    | MediaPlayerEntityFeature.VOLUME_MUTE
)

AREAS: list[dict[str, Any]] = [
    {"area_id": "area_living", "name": "Living Room", "aliases": [], "picture": None},
    {"area_id": "area_kitchen", "name": "Kitchen", "aliases": [], "picture": None},
    {"area_id": "area_office", "name": "Office", "aliases": [], "picture": None},
]

DEVICES: list[dict[str, Any]] = [
    {"id": "dev_living", "name": "Living Room Amp", "name_by_user": None, "area_id": "area_living"},
    {
        "id": "dev_kitchen",
        "name": "Speaker",
        "name_by_user": "Kitchen Speaker",
        "area_id": "area_kitchen",
    },
    {"id": "dev_attic", "name": "Attic Box", "name_by_user": None, "area_id": None},
]

# entity_id -> (device_id, area_id override, friendly_name, extra state attributes)
ENTITIES: dict[str, tuple[str | None, str | None, str, dict[str, Any]]] = {
    "media_player.living_amp": (
        "dev_living",
        None,
        "Amplifier",
        {"supported_features": FULL_MEDIA_PLAYER_FEATURES},
    ),
    "media_player.mass_player": (
        "dev_living",
        None,
        "MA Player",
        {"supported_features": FULL_MEDIA_PLAYER_FEATURES, "mass_player_type": "player"},
    ),
    "media_player.featureless": (None, None, "Featureless Player", {"supported_features": 0}),
    "switch.kitchen_power": ("dev_kitchen", None, "Kitchen Power", {}),
    "number.kitchen_volume": ("dev_kitchen", "area_office", "Kitchen Volume", {}),
    "input_boolean.standalone": (None, "area_living", "Standalone Toggle", {}),
    "input_number.attic_volume": ("dev_attic", None, "Attic Volume", {}),
    # not a control domain at all, so it must never surface
    "light.hallway": (None, "area_living", "Hallway Light", {}),
}


def _config() -> MagicMock:
    """Return a provider config exposing the persisted values via get_value."""
    persisted_values: dict[str, Any] = {
        CONF_URL: "http://homeassistant.local:8123",
        CONF_AUTH_TOKEN: "token",
        CONF_VERIFY_SSL: True,
        CONF_LOG_LEVEL: "GLOBAL",
        CONF_POWER_CONTROLS: [],
        CONF_MUTE_CONTROLS: [],
        CONF_VOLUME_CONTROLS: [],
    }
    config = MagicMock()
    config.instance_id = "hass--test"
    config.name = "Home Assistant"
    config.get_value.side_effect = persisted_values.get
    config.values = {}
    return config


def _mass() -> MagicMock:
    """Return the Music Assistant dependencies used during provider startup."""
    mass = MagicMock()
    mass.cache = MagicMock()
    mass.http_session = MagicMock()
    mass.http_session_no_ssl = MagicMock()
    mass.create_task.side_effect = asyncio.create_task
    mass.players.register_or_update_player_control = AsyncMock()
    mass.config.get = MagicMock(return_value={})
    mass.config.get_raw_provider_config_value = MagicMock(return_value=None)
    return mass


class _HomeAssistantClient:
    """Serve the registries and states of a small Home Assistant install."""

    def __init__(self) -> None:
        self.connected = False
        self.listener_started = asyncio.Event()
        # entity_ids per subscribe_entities call, in call order
        self.state_requests: list[list[str]] = []
        self.registry_list_calls = 0
        self.device_registry_calls = 0
        self.area_registry_calls = 0
        self.event_subscriptions: list[tuple[str, Callable[[dict[str, Any]], None]]] = []
        self.send_command = AsyncMock(side_effect=self._send_command)

    async def connect(self) -> None:
        """Connect the client."""
        self.connected = True

    async def start_listening(self) -> None:
        """Listen until the provider stops the listener task."""
        self.listener_started.set()
        await asyncio.Event().wait()

    async def disconnect(self) -> None:
        """Disconnect the client."""
        self.connected = False

    async def get_device_registry(self) -> list[dict[str, Any]]:
        """Return the device registry listing."""
        self.device_registry_calls += 1
        return DEVICES

    async def get_area_registry(self) -> list[dict[str, Any]]:
        """Return the area registry listing."""
        self.area_registry_calls += 1
        return AREAS

    async def subscribe_entities(
        self, cb_func: Callable[[dict[str, Any]], None], entity_ids: list[str]
    ) -> Callable[[], None]:
        """Deliver the requested states and return the unsubscribe callable."""
        self.state_requests.append(list(entity_ids))
        initial = {
            entity_id: {"s": "idle", "a": _attributes(entity_id)}
            for entity_id in entity_ids
            if entity_id in ENTITIES
        }
        asyncio.get_running_loop().call_soon(cb_func, {"a": initial})
        return lambda: None

    async def subscribe_events(
        self, cb_func: Callable[[dict[str, Any]], None], event_type: str
    ) -> Callable[[], None]:
        """Register the event callback after command responses can be received."""
        await self.listener_started.wait()
        self.event_subscriptions.append((event_type, cb_func))
        return lambda: None

    def fire_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Deliver an event to every subscriber of the given event type."""
        for subscribed_type, cb_func in self.event_subscriptions:
            if subscribed_type == event_type:
                cb_func({"event_type": event_type, "data": data})

    async def _send_command(self, command: str, **kwargs: Any) -> Any:
        """Return the response Home Assistant sends for the given websocket command."""
        if command != REGISTRY_LIST_COMMAND:
            return {}
        await self.listener_started.wait()
        self.registry_list_calls += 1
        return {
            "entity_categories": {},
            "entities": [
                {"ei": entity_id, "pl": "test", "di": device_id, "ai": area_id}
                for entity_id, (device_id, area_id, _, _) in ENTITIES.items()
            ],
        }


def _attributes(entity_id: str) -> dict[str, Any]:
    """Return the state attributes of the given entity."""
    _, _, friendly_name, extra = ENTITIES[entity_id]
    return {"friendly_name": friendly_name, **extra}


def _entity_ids(groups: list[HassControlEntityGroup]) -> list[str]:
    """Return the entity IDs of all groups, in result order."""
    return [entity["entity_id"] for group in groups for entity in group["entities"]]


@asynccontextmanager
async def _start_provider() -> AsyncIterator[tuple[HomeAssistantProvider, _HomeAssistantClient]]:
    """Start the provider with a connected mocked Home Assistant client."""
    hass = _HomeAssistantClient()
    manifest = MagicMock()
    manifest.domain = "hass"
    manifest.name = "Home Assistant"
    with patch("music_assistant.providers.hass.HomeAssistantClient", return_value=hass):
        provider = await setup(_mass(), manifest, _config())
        assert isinstance(provider, HomeAssistantProvider)
        async with asyncio.timeout(5):
            await provider.handle_async_init()
    try:
        yield provider, hass
    finally:
        await provider.unload()


async def test_search_command_is_exposed_on_the_api() -> None:
    """Expose the search as a read-only API command for as long as the provider is loaded."""
    async with _start_provider() as (provider, _):
        mass = cast("MagicMock", provider.mass)
        unregister = mass.register_api_command.return_value
        await provider.loaded_in_mass()
        mass.register_api_command.assert_called_once_with(
            SEARCH_CONTROL_ENTITIES_COMMAND,
            provider.search_control_entities,
            required_scope=Scope.CONFIG_PROVIDERS_READ,
        )
        # the API layer must be able to resolve the command's signature and result type
        handler = APICommandHandler.parse(
            SEARCH_CONTROL_ENTITIES_COMMAND, provider.search_control_entities
        )
        assert handler.type_hints["return"] is HassControlEntitySearchResult
        unregister.assert_not_called()

    unregister.assert_called_once_with()


@pytest.mark.parametrize(
    ("search", "expected"),
    [
        # entity id
        ("kitchen_power", ["switch.kitchen_power"]),
        # friendly name
        ("amplifier", ["media_player.living_amp"]),
        # device name (name_by_user wins over name)
        ("kitchen speaker", ["switch.kitchen_power", "number.kitchen_volume"]),
        # area name, inherited from the device
        ("living room", ["media_player.living_amp", "input_boolean.standalone"]),
        # area name of an entity that overrides its device's area
        ("office", ["number.kitchen_volume"]),
    ],
)
async def test_search_matches_every_searchable_field(search: str, expected: list[str]) -> None:
    """Match the search text against entity ID, entity name, device name and area name."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities(search=search)

    assert sorted(_entity_ids(result["groups"])) == sorted(expected)
    assert result["truncated"] is False


async def test_search_is_case_insensitive() -> None:
    """Match regardless of the casing of the search text."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities(search="AMPLIFIER")

    assert _entity_ids(result["groups"]) == ["media_player.living_amp"]


async def test_search_without_text_returns_all_eligible_entities() -> None:
    """Return every entity that can serve a control role when no search text is given."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities()

    assert sorted(_entity_ids(result["groups"])) == [
        "input_boolean.standalone",
        "input_number.attic_volume",
        "media_player.living_amp",
        "number.kitchen_volume",
        "switch.kitchen_power",
    ]


async def test_search_excludes_music_assistant_players() -> None:
    """Never offer Music Assistant's own exposed players as a control."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities(search="player")

    assert "media_player.mass_player" not in _entity_ids(result["groups"])
    # the entity without any usable feature is left out too
    assert "media_player.featureless" not in _entity_ids(result["groups"])


async def test_control_type_filters_on_capability() -> None:
    """Return only the entities that can serve the requested control role."""
    async with _start_provider() as (provider, hass):
        hass.state_requests.clear()
        volume = await provider.search_control_entities(control_type=CONF_VOLUME_CONTROLS)
        volume_requests = [entity_id for request in hass.state_requests for entity_id in request]
        power = await provider.search_control_entities(control_type=CONF_POWER_CONTROLS)
        mute = await provider.search_control_entities(control_type=CONF_MUTE_CONTROLS)

    assert sorted(_entity_ids(volume["groups"])) == [
        "input_number.attic_volume",
        "media_player.living_amp",
        "number.kitchen_volume",
    ]
    assert sorted(_entity_ids(power["groups"])) == [
        "input_boolean.standalone",
        "media_player.living_amp",
        "switch.kitchen_power",
    ]
    assert sorted(_entity_ids(mute["groups"])) == [
        "input_boolean.standalone",
        "media_player.living_amp",
        "switch.kitchen_power",
    ]
    # a volume search must not sweep the states of the on/off-only domains
    assert volume_requests == [
        "input_number.attic_volume",
        "media_player.featureless",
        "media_player.living_amp",
        "media_player.mass_player",
        "number.kitchen_volume",
    ]


async def test_control_type_reports_every_supported_role() -> None:
    """Report all roles an entity can serve, not just the one that was searched for."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities(
            search="kitchen_power", control_type=CONF_POWER_CONTROLS
        )

    entity = result["groups"][0]["entities"][0]
    assert (entity["power"], entity["volume"], entity["mute"]) == (True, False, True)


async def test_unknown_control_type_is_rejected() -> None:
    """Reject a control type that does not exist instead of returning everything."""
    async with _start_provider() as (provider, _):
        with pytest.raises(InvalidDataError, match="Invalid control type"):
            await provider.search_control_entities(control_type="brightness_controls")


async def test_results_are_grouped_by_device_and_area() -> None:
    """Group the entities by device and effective area, ordered by area, device and name."""
    async with _start_provider() as (provider, _):
        result = await provider.search_control_entities()

    assert [
        (group["device_id"], group["device_name"], group["area_name"], _entity_ids([group]))
        for group in result["groups"]
    ] == [
        ("dev_kitchen", "Kitchen Speaker", "Kitchen", ["switch.kitchen_power"]),
        # the entity inherits the area of its device
        ("dev_living", "Living Room Amp", "Living Room", ["media_player.living_amp"]),
        # an entity without a device falls back to a group of its own area
        (None, None, "Living Room", ["input_boolean.standalone"]),
        # the entity overrides the area it would inherit from its (Kitchen) device
        ("dev_kitchen", "Kitchen Speaker", "Office", ["number.kitchen_volume"]),
        # a device without an area sorts last
        ("dev_attic", "Attic Box", None, ["input_number.attic_volume"]),
    ]


async def test_limit_caps_entities_and_reports_truncation() -> None:
    """Cap the number of entities and tell the caller that matches were left out."""
    async with _start_provider() as (provider, _):
        limited = await provider.search_control_entities(limit=2)
        exact = await provider.search_control_entities(limit=5)

    assert _entity_ids(limited["groups"]) == ["switch.kitchen_power", "media_player.living_amp"]
    assert limited["truncated"] is True
    assert len(_entity_ids(exact["groups"])) == 5
    assert exact["truncated"] is False


async def test_consecutive_searches_share_one_state_sweep() -> None:
    """Sweep the Home Assistant states once and serve the next search from the cache."""
    async with _start_provider() as (provider, hass):
        hass.state_requests.clear()
        await provider.search_control_entities(search="living")
        await provider.search_control_entities(search="living room")
        sweeps = len(hass.state_requests)

    assert sweeps == 1


async def test_concurrent_searches_share_one_state_sweep() -> None:
    """Let searches that arrive together wait for a single state sweep."""
    async with _start_provider() as (provider, hass):
        hass.state_requests.clear()
        results = await asyncio.gather(
            provider.search_control_entities(search="kitchen"),
            provider.search_control_entities(search="living"),
        )
        sweeps = len(hass.state_requests)

    assert sweeps == 1
    assert _entity_ids(results[0]["groups"]) == [
        "switch.kitchen_power",
        "number.kitchen_volume",
    ]


async def test_power_and_mute_searches_share_one_state_sweep() -> None:
    """Reuse one sweep for the control roles that live in the same entity domains."""
    async with _start_provider() as (provider, hass):
        hass.state_requests.clear()
        await provider.search_control_entities(control_type=CONF_POWER_CONTROLS)
        await provider.search_control_entities(control_type=CONF_MUTE_CONTROLS)
        after_power_and_mute = len(hass.state_requests)
        await provider.search_control_entities(control_type=CONF_VOLUME_CONTROLS)
        after_volume = len(hass.state_requests)

    assert after_power_and_mute == 1
    # the volume roles live in other domains, so they need a sweep of their own
    assert after_volume == 2


async def test_entity_registry_change_forces_a_fresh_state_sweep() -> None:
    """Sweep again after Home Assistant reports an entity registry change."""
    async with _start_provider() as (provider, hass):
        hass.state_requests.clear()
        await provider.search_control_entities()
        hass.fire_event(
            "entity_registry_updated",
            {"action": "update", "entity_id": "media_player.living_amp"},
        )
        await provider.search_control_entities()
        sweeps = len(hass.state_requests)

    assert sweeps == 2


async def test_cached_candidates_expire() -> None:
    """Sweep again once the cached candidates have outlived their TTL."""
    with patch("music_assistant.providers.hass.control_entities.CONTROL_ENTITY_CACHE_TTL", 0):
        async with _start_provider() as (provider, hass):
            hass.state_requests.clear()
            await provider.search_control_entities()
            await provider.search_control_entities()
            sweeps = len(hass.state_requests)

    assert sweeps == 2


async def test_search_reuses_the_cached_registries() -> None:
    """Consult Home Assistant for the registries once and reuse them on the next search."""
    async with _start_provider() as (provider, hass):
        registry_calls_after_startup = hass.registry_list_calls
        await provider.search_control_entities()
        after_first = (
            hass.registry_list_calls,
            hass.device_registry_calls,
            hass.area_registry_calls,
        )
        await provider.search_control_entities(search="kitchen")
        after_second = (
            hass.registry_list_calls,
            hass.device_registry_calls,
            hass.area_registry_calls,
        )

    assert registry_calls_after_startup == 1
    assert after_first == (1, 1, 1)
    assert after_second == (1, 1, 1)


async def test_device_and_area_registry_updates_invalidate_the_cache() -> None:
    """Rebuild the candidates from scratch after a device or area registry change."""
    async with _start_provider() as (provider, hass):
        await provider.search_control_entities()
        hass.state_requests.clear()
        hass.fire_event("device_registry_updated", {"action": "update", "device_id": "dev_living"})
        await provider.search_control_entities()
        after_device_change = (
            hass.device_registry_calls,
            hass.area_registry_calls,
            len(hass.state_requests),
        )
        hass.fire_event("area_registry_updated", {"action": "update", "area_id": "area_living"})
        await provider.search_control_entities()
        after_area_change = (
            hass.device_registry_calls,
            hass.area_registry_calls,
            len(hass.state_requests),
        )

    # a renamed device must not stay hidden behind the cached candidates until the TTL passes
    assert after_device_change == (2, 1, 1)
    assert after_area_change == (2, 2, 2)
