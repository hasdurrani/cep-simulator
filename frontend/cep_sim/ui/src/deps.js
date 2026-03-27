// Centralised CDN imports — all components import from here
// Pinned to exact versions so esm.sh can deduplicate correctly (react-dom must
// use the same react instance as the components).
import {
  createElement,
  Component,
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "https://esm.sh/react@18.3.1?dev";
export { createRoot } from "https://esm.sh/react-dom@18.3.1/client?dev";
export { default as htm } from "https://esm.sh/htm@3.1.1";

export { Component, useState, useEffect, useCallback, useRef, useMemo };

// htm uses `class` in templates; React DOM requires `className`.
// This wrapper converts once so no component needs to use className.
export function h(type, props, ...children) {
  if (props && typeof type === "string" && props.class !== undefined) {
    const { class: cls, ...rest } = props;
    props = { ...rest, className: cls };
  }
  return createElement(type, props, ...children);
}
