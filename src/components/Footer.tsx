// src/components/Footer.tsx
import React from 'react';
import { Box, Typography, IconButton } from '@mui/material';
import { FaInstagram } from 'react-icons/fa';

const Footer = () => (
  <Box
    component="footer"
    sx={{
      textAlign: 'center',
      padding: '20px',
      background: '#2C2C3E',
      color: '#FFFFFF',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
    }}
  >
    <Typography variant="body2">
      © Neurolink {new Date().getFullYear()}
    </Typography>
    <Box>
      <IconButton href="https://www.instagram.com" target="_blank" rel="noopener" sx={{ color: '#FFFFFF' }}>
        <FaInstagram />
      </IconButton>
      {/* Add more social icons as needed */}
    </Box>
    <Typography variant="body2">
      Additional footer content here.
    </Typography>
  </Box>
);

export default Footer;
