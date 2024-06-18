// src/components/GettingStarted.tsx
import React from 'react';
import { Box, Container, Typography } from '@mui/material';

const GettingStarted = () => (
  <Box
    sx={{
      padding: '50px 0',
      minHeight: 'calc(100vh - 128px)', // Adjust height to account for header and footer
    }}
  >
    <Container>

      <Box
        sx={{
          position: 'relative',
          paddingTop: '56.25%', // 16:9 aspect ratio
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
        }}
      >
        <iframe
          src="https://www.youtube.com/embed/dQw4w9WgXcQ"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          title="video"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
          }}
        />
      </Box>
    </Container>
  </Box>
);

export default GettingStarted;
