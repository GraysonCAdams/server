"""Searchable listing of the Home Assistant entities that can act as a player control."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Final, NamedTuple, TypedDict

from music_assistant_models.errors import InvalidDataError

from .constants import (
    CONF_MUTE_CONTROLS,
    CONF_POWER_CONTROLS,
    CONF_VOLUME_CONTROLS,
    CONTROL_DOMAINS,
)
from .helpers import get_control_capabilities

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from hass_client.models import Area, Device, Event

    from . import HomeAssistantProvider

SEARCH_CONTROL_ENTITIES_LIMIT = 50
# How long a search may reuse the control entity candidates of an earlier search. Resolving
# them takes a full state sweep of Home Assistant, so a picker that searches while the user
# types would otherwise sweep on every keystroke. The candidates carry entity, device and
# area names plus control capabilities - none of which change often, and none of which is
# live entity state - so briefly serving a stale listing is harmless. A change to any of the
# registries they build on drops the candidates right away rather than waiting this out.
CONTROL_ENTITY_CACHE_TTL = 30

# The entity domains worth inspecting for each control role. A superset is harmless: the
# authoritative verdict comes from get_control_capabilities on the entity's own state,
# this only keeps the state sweep of a search away from domains that can never qualify.
CONTROL_TYPE_DOMAINS: Final[dict[str, tuple[str, ...]]] = {
    CONF_POWER_CONTROLS: ("media_player", "switch", "input_boolean"),
    CONF_VOLUME_CONTROLS: ("media_player", "number", "input_number"),
    CONF_MUTE_CONTROLS: ("media_player", "switch", "input_boolean"),
}


class HassControlEntity(TypedDict):
    """A Home Assistant entity that can be used as a player control."""

    entity_id: str
    # the entity's friendly name, falling back to its entity ID when it has none
    name: str
    power: bool
    volume: bool
    mute: bool


# Selects the entities that can serve each control role.
CONTROL_TYPE_CAPABILITIES: Final[dict[str, Callable[[HassControlEntity], bool]]] = {
    CONF_POWER_CONTROLS: lambda entity: entity["power"],
    CONF_VOLUME_CONTROLS: lambda entity: entity["volume"],
    CONF_MUTE_CONTROLS: lambda entity: entity["mute"],
}


class HassControlEntityGroup(TypedDict):
    """The control entities of a single Home Assistant device."""

    device_id: str | None
    # None for entities that belong to no device, respectively to no area
    device_name: str | None
    area_name: str | None
    entities: list[HassControlEntity]


class HassControlEntitySearchResult(TypedDict):
    """The outcome of a player control entity search."""

    groups: list[HassControlEntityGroup]
    # True when matches were left out to honor the requested limit
    truncated: bool


class ControlEntitySearch:
    """Searchable view on the Home Assistant entities that can act as a player control."""

    def __init__(self, provider: HomeAssistantProvider) -> None:
        """
        Initialize the search.

        :param provider: The Home Assistant provider to read the registries and states from;
            its client must be connected before the first search.
        """
        self._provider = provider
        self._device_registry: _RegistryCache[Device] = _RegistryCache(
            "device_registry_updated",
            self._fetch_device_registry,
            provider.hass.subscribe_events,
        )
        self._area_registry: _RegistryCache[Area] = _RegistryCache(
            "area_registry_updated",
            self._fetch_area_registry,
            provider.hass.subscribe_events,
        )
        self._candidates = _ControlEntityCache(
            CONTROL_ENTITY_CACHE_TTL, self._generations, self._resolve
        )

    async def search(
        self,
        search: str | None = None,
        control_type: str | None = None,
        limit: int = SEARCH_CONTROL_ENTITIES_LIMIT,
    ) -> HassControlEntitySearchResult:
        """
        Search the Home Assistant entities that can be used as a player control.

        Music Assistant's own players are never part of the result. Consecutive searches are
        served from a short lived cache that any entity, device or area registry change drops
        right away, so only a name or capability change Home Assistant makes without touching
        a registry can lag by a few seconds.

        :param search: Text to match, case insensitively, against the entity ID, the entity
            name, its device name and its area name. All eligible entities match when omitted.
        :param control_type: Restrict the result to entities that can serve this control role,
            given as one of the provider's control config keys (``power_controls``,
            ``volume_controls`` or ``mute_controls``). All roles are returned when omitted.
        :param limit: Maximum number of entities (not groups) to return.
        :return: The matching entities grouped by the device and area they belong to, ordered
            by area, device and entity name, plus a flag telling whether matches were left out
            to honor the limit.
        """
        if control_type is not None and control_type not in CONTROL_TYPE_DOMAINS:
            msg = f"Invalid control type: {control_type}"
            raise InvalidDataError(msg)
        domains = CONTROL_DOMAINS if control_type is None else CONTROL_TYPE_DOMAINS[control_type]
        matches = await self._candidates.get(domains)
        if control_type is not None:
            has_capability = CONTROL_TYPE_CAPABILITIES[control_type]
            matches = [match for match in matches if has_capability(match.entity)]
        if query := (search or "").casefold():
            matches = [match for match in matches if match.matches(query)]
        limit = max(limit, 0)
        groups: dict[tuple[str | None, str | None], HassControlEntityGroup] = {}
        for match in matches[:limit]:
            group_key = (match.device_id, match.area_id)
            if (group := groups.get(group_key)) is None:
                group = HassControlEntityGroup(
                    device_id=match.device_id,
                    device_name=match.device_name,
                    area_name=match.area_name,
                    entities=[],
                )
                groups[group_key] = group
            group["entities"].append(match.entity)
        return HassControlEntitySearchResult(
            groups=list(groups.values()), truncated=len(matches) > limit
        )

    def close(self) -> None:
        """Drop everything cached and stop watching for registry updates."""
        self._device_registry.close()
        self._area_registry.close()
        self._candidates.clear()

    def _generations(self) -> tuple[int, ...]:
        """Return the generations of the registries the control entity candidates build on."""
        return (
            self._provider.entity_registry_generation,
            self._device_registry.generation,
            self._area_registry.generation,
        )

    async def _resolve(self, domains: tuple[str, ...]) -> list[_ControlEntityMatch]:
        """
        Return the entities of the given domains that can serve as a player control.

        :param domains: The entity domains to consider.
        :return: The candidates in presentation order, each carrying every control role it
            can serve plus the device and area it belongs to.
        """
        entity_registry = await self._provider.get_entity_registry()
        devices = await self._device_registry.get()
        areas = await self._area_registry.get()
        matches: list[_ControlEntityMatch] = []
        for state in await self._provider.get_states(domains=domains):
            capabilities = get_control_capabilities(state, self._provider.logger)
            if not any(capabilities):
                continue
            entity_id = state["entity_id"]
            registry_entry = entity_registry.get(entity_id)
            device_id = registry_entry["device_id"] if registry_entry else None
            device = devices.get(device_id) if device_id else None
            # Home Assistant lets an entity override the area it inherits from its device
            area_id = (registry_entry["area_id"] if registry_entry else None) or (
                device["area_id"] if device else None
            )
            matches.append(
                _ControlEntityMatch(
                    device_id=device_id,
                    device_name=(device["name_by_user"] or device["name"]) if device else None,
                    area_id=area_id,
                    area_name=area["name"] if (area := areas.get(area_id or "")) else None,
                    entity=HassControlEntity(
                        entity_id=entity_id,
                        name=state["attributes"].get("friendly_name") or entity_id,
                        power=capabilities.power,
                        volume=capabilities.volume,
                        mute=capabilities.mute,
                    ),
                )
            )
        matches.sort(key=lambda match: match.sort_key)
        return matches

    async def _fetch_device_registry(self) -> dict[str, Device]:
        """Fetch the device registry from Home Assistant, keyed by device ID."""
        return {device["id"]: device for device in await self._provider.hass.get_device_registry()}

    async def _fetch_area_registry(self) -> dict[str, Area]:
        """Fetch the area registry from Home Assistant, keyed by area ID."""
        return {area["area_id"]: area for area in await self._provider.hass.get_area_registry()}


class _ControlEntityMatch(NamedTuple):
    """A control entity together with the device and area it is presented under."""

    device_id: str | None
    device_name: str | None
    area_id: str | None
    area_name: str | None
    entity: HassControlEntity

    @property
    def sort_key(self) -> tuple[bool, str, bool, str, str, str]:
        """Return the ranking key, placing entities without an area or device last."""
        return (
            self.area_name is None,
            (self.area_name or "").casefold(),
            self.device_name is None,
            (self.device_name or "").casefold(),
            self.entity["name"].casefold(),
            self.entity["entity_id"],
        )

    def matches(self, query: str) -> bool:
        """
        Return whether the given search text occurs in any of the searchable fields.

        :param query: The case folded search text.
        """
        return any(
            query in field.casefold()
            for field in (
                self.entity["entity_id"],
                self.entity["name"],
                self.device_name,
                self.area_name,
            )
            if field
        )


class _RegistryCache[ItemT]:
    """Cached Home Assistant registry listing that refreshes on the registry's update event."""

    def __init__(
        self,
        event_type: str,
        fetch: Callable[[], Awaitable[dict[str, ItemT]]],
        subscribe: Callable[[Callable[[Event], None], str], Awaitable[Callable[[], None]]],
    ) -> None:
        """
        Initialize the cache.

        :param event_type: The Home Assistant event that signals a change to this registry.
        :param fetch: Coroutine function returning the registry listing keyed by item ID.
        :param subscribe: The Home Assistant client's event subscription method.
        """
        self._event_type = event_type
        self._fetch = fetch
        self._subscribe = subscribe
        self._lock = asyncio.Lock()
        self._items: dict[str, ItemT] | None = None
        self._generation = 0
        self._unsubscribe: Callable[[], None] | None = None

    @property
    def generation(self) -> int:
        """Return a counter that changes whenever the cached listing is invalidated."""
        return self._generation

    async def get(self) -> dict[str, ItemT]:
        """Return the registry listing, keyed by item ID."""
        if (items := self._items) is not None:
            return items
        async with self._lock:
            if self._unsubscribe is None:
                # the subscription must be live before the first read, so no registry
                # change can slip through unnoticed
                self._unsubscribe = await self._subscribe(self._invalidate, self._event_type)
            if (items := self._items) is None:
                generation = self._generation
                items = await self._fetch()
                # a registry change while the fetch was in flight leaves the listing stale
                # on arrival, so serve it to this caller but keep it out of the cache
                if generation == self._generation:
                    self._items = items
            return items

    def close(self) -> None:
        """Drop the cached listing and stop watching for registry updates."""
        if unsubscribe := self._unsubscribe:
            self._unsubscribe = None
            unsubscribe()
        self._invalidate()

    def _invalidate(self, _event: Event | None = None) -> None:
        """Drop the cached listing."""
        self._items = None
        self._generation += 1


class _ControlEntityCacheEntry(NamedTuple):
    """Cached control entity candidates together with what makes them go stale."""

    expires_at: float
    generations: tuple[int, ...]
    matches: list[_ControlEntityMatch]


class _ControlEntityCache:
    """
    Short lived cache of the player control candidates, per set of entity domains.

    Resolving the candidates takes a full state sweep of Home Assistant, by far the most
    expensive part of a search, so consecutive searches reuse a single sweep.
    """

    def __init__(
        self,
        ttl: float,
        generations: Callable[[], tuple[int, ...]],
        resolve: Callable[[tuple[str, ...]], Awaitable[list[_ControlEntityMatch]]],
    ) -> None:
        """
        Initialize the cache.

        :param ttl: How long resolved candidates may be reused, in seconds.
        :param generations: Returns the generations of the registries the candidates are built
            from, so candidates resolved against an outdated registry are discarded.
        :param resolve: Coroutine function resolving the candidates of the given domains.
        """
        self._ttl = ttl
        self._generations = generations
        self._resolve = resolve
        self._lock = asyncio.Lock()
        self._entries: dict[tuple[str, ...], _ControlEntityCacheEntry] = {}

    async def get(self, domains: tuple[str, ...]) -> list[_ControlEntityMatch]:
        """
        Return the player control candidates found in the given entity domains.

        :param domains: The entity domains to consider.
        """
        if (matches := self._lookup(domains)) is not None:
            return matches
        async with self._lock:
            # a concurrent search may have resolved the same domains while this one waited
            if (matches := self._lookup(domains)) is not None:
                return matches
            generations = self._generations()
            matches = await self._resolve(domains)
            # a registry change while the sweep was in flight leaves the candidates stale on
            # arrival, so serve them to this caller but keep them out of the cache
            if generations == self._generations():
                self._entries[domains] = _ControlEntityCacheEntry(
                    expires_at=time.monotonic() + self._ttl,
                    generations=generations,
                    matches=matches,
                )
            return matches

    def clear(self) -> None:
        """Drop all cached candidates."""
        self._entries.clear()

    def _lookup(self, domains: tuple[str, ...]) -> list[_ControlEntityMatch] | None:
        """Return the cached candidates of the given domains, None when there are none left."""
        if (entry := self._entries.get(domains)) is None:
            return None
        if entry.generations != self._generations() or entry.expires_at <= time.monotonic():
            del self._entries[domains]
            return None
        return entry.matches
