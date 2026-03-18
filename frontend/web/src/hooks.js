import { useEffect, useState } from "react";

export function useAsyncResource(loader, deps) {
  const [state, setState] = useState({
    data: null,
    error: null,
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, error: null, loading: true });

    loader()
      .then((data) => {
        if (!cancelled) {
          setState({ data, error: null, loading: false });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setState({ data: null, error, loading: false });
        }
      });

    return () => {
      cancelled = true;
    };
  }, deps);

  return state;
}
